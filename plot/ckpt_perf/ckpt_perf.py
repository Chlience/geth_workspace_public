"""
Script to benchmark checkpoint save/load time for model and optimizer.
Usage: python ckpt_perf.py --repeats 5
"""
import time
import io
import os
import sys
import argparse
import tempfile
import torch
import importlib
import types
import torch.distributed as dist
import torch.multiprocessing as mp
import pickle
import tempfile as _tempfile

# optimizer state init per base_trainer
from torch import optim

def module_from_file(module_name, file_path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"load client module {module_name} from file failed!"
    assert spec.loader is not None, (
        f"load client module {module_name} from file failed!"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"load client module {module} from file success!")
    return module

def init_optimizer_state(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state[p] = {}
            # Adam
            if isinstance(optimizer, optim.Adam):
                optimizer.state[p]['step'] = torch.tensor(0.0, device=p.device)
                optimizer.state[p]['exp_avg'] = torch.zeros_like(p.data, device=p.device)
                optimizer.state[p]['exp_avg_sq'] = torch.zeros_like(p.data, device=p.device)
            # RMSprop
            if isinstance(optimizer, optim.RMSprop) and group.get('momentum', 0) > 0:
                optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p.data, device=p.device)
            # SGD
            if isinstance(optimizer, optim.SGD) and group.get('momentum', 0) > 0:
                optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p.data, device=p.device)
            if isinstance(optimizer, optim.SGD) and group.get('nesterov', False):
                optimizer.state[p]['velocity'] = torch.zeros_like(p.data, device=p.device)

# function to count model and optimizer parameters
def count_model_optimizer_params(model, optimizer):
    """
    Compute number of parameters in the model and in the optimizer state.
    Returns (model_elem_count, model_tensor_count, opt_elem_count, opt_tensor_count).
    """
    # total model elements and tensor count
    model_elem_count = sum(p.numel() for p in model.parameters())
    model_tensor_count = sum(1 for _ in model.parameters())
    # total optimizer state elements and tensor count
    opt_state = optimizer.state_dict()
    opt_elem_count = 0
    opt_tensor_count = 0
    for state in opt_state.get('state', {}).values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                opt_elem_count += v.numel()
                opt_tensor_count += 1
    return model_elem_count, model_tensor_count, opt_elem_count, opt_tensor_count

# comparative raw tensor broadcast: direct model & optimizer state dict
def raw_broadcast_worker(rank, world_size, repeats, init_method, task_name, config_file):
    torch.cuda.set_device(rank)
    # init process group
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    # load model and optimizer on each rank
    cfg = module_from_file(task_name, config_file).task_config
    model = cfg.get_model().to(torch.device(rank)); opt = cfg.get_optimizer()
    init_optimizer_state(opt)
    times = []
    for _ in range(repeats):
        dist.barrier()
        # collect tensors to send/receive in batches
        model_tensors = list(model.state_dict().values())
        opt_tensors = []
        for state in opt.state.values():
            opt_tensors.extend(list(state.values()))
        t0 = time.time()
        
        ops = []
        if rank == 0:  # sender
            # create P2POp for model parameters and optimizer state
            for dst in range(1, world_size):
                for tensor in model_tensors:
                    ops.append(dist.P2POp(dist.isend, tensor, dst))
                for tensor in opt_tensors:
                    ops.append(dist.P2POp(dist.isend, tensor, dst))
        else:  # receivers
            # create P2POp for model parameters and optimizer state
            for tensor in model_tensors:
                ops.append(dist.P2POp(dist.irecv, tensor, 0))
            for tensor in opt_tensors:
                ops.append(dist.P2POp(dist.irecv, tensor, 0))
        
        # execute all operations
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        torch.cuda.synchronize()
        
        times.append(time.time() - t0)
        dist.barrier()
    if rank == 0:
        print(f"[RawBroadcast] world_size={world_size}, avg={sum(times)/len(times):.6f}s, times={times}")
    dist.destroy_process_group()


# comparative benchmark: broadcast checkpoint to multiple processes
# helper worker for broadcast
def broadcast_worker(rank, world_size, repeats, init_method, task_name, config_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(init_method=init_method, rank=rank, world_size=world_size)
    # load model and optimizer on each rank
    cfg = module_from_file(task_name, config_file).task_config
    model = cfg.get_model(); opt = cfg.get_optimizer()
    init_optimizer_state(opt)
    times = []
    for _ in range(repeats):
        dist.barrier()
        t0 = time.time()
        if rank == 0:
            # Serialize model and optimizer state to buffer
            buffer = io.BytesIO()
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, buffer)
            buffer.seek(0)
            buf = buffer.read()
            obj_list = [buf]
        else:
            obj_list = [None]
        
        # Broadcast size first
        dist.broadcast_object_list(obj_list, src=0)

        # deserialize and load checkpoint
        if isinstance(obj_list[0], bytes):
            buffer = io.BytesIO(obj_list[0])
            d = torch.load(buffer, map_location=torch.device(rank))
        else:
            d = torch.load(obj_list[0], map_location=torch.device(rank))
        model.load_state_dict(d['model'])
        opt.load_state_dict(d['optimizer'])

        times.append(time.time() - t0)
        dist.barrier()

    if rank == 0:
        print(f"[BroadcastSerialized] world_size={world_size}, avg={sum(times)/len(times):.6f}s, times={times}")
    dist.destroy_process_group()

# define worker function for checkpoint benchmarking
def ckpt_worker(rank, world_size, repeats, init_method, task_name, config_file):
    torch.cuda.set_device(rank)
    # load task config dynamically if provided
    if task_name and config_file:
        config_module = module_from_file(task_name, config_file)
        task_config = config_module.task_config
    else:
        assert False
    model = task_config.get_model()
    optimizer = task_config.get_optimizer()
    # init optimizer state
    init_optimizer_state(optimizer)

    # count model and optimizer parameters (elements and tensors)
    model_elem_count, model_tensor_count, opt_elem_count, opt_tensor_count = count_model_optimizer_params(model, optimizer)
    print(f"[ModelParams] model_elem_count={model_elem_count}, model_tensor_count={model_tensor_count}, opt_elem_count={opt_elem_count}, opt_tensor_count={opt_tensor_count}")

    save_times = []
    load_times = []

    for _ in range(repeats):
        # create temp file
        path = "ckpt.pt"
        # save
        t0 = time.time()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
        # flush file
        os.sync()
        save_times.append(time.time() - t0)
        # load
        t1 = time.time()
        d = torch.load(path, map_location=torch.device(rank))
        model.load_state_dict(d['model'])
        optimizer.load_state_dict(d['optimizer'])
        load_times.append(time.time() - t1)
        # remove
        os.remove(path)

    print(f"[SaveCkpt] avg={sum(save_times)/len(save_times):.6f}s, times={save_times}")
    print(f"[LoadCkpt] avg={sum(load_times)/len(load_times):.6f}s, times={load_times}")

def benchmark(repeats, task_name=None, config_file=None):
    # spawn checkpoint save/load tests
    with _tempfile.NamedTemporaryFile(delete=False) as tmpf:
        init_method = f"file://{tmpf.name}"
    mp.spawn(ckpt_worker, args=(1, repeats, init_method, task_name, config_file), nprocs=1, join=True)

    # test broadcast for processes counts 2,4,6,8
    for world_size in [2, 4, 6, 8]:
        # create unique init file
        with _tempfile.NamedTemporaryFile(delete=False) as tmpf:
            init_method = f"file://{tmpf.name}"
        mp.spawn(broadcast_worker, args=(world_size, repeats, init_method, task_name, config_file), nprocs=world_size, join=True)

    
    # spawn raw tensor broadcast tests
    for world_size in [2, 4, 6, 8]:
        with _tempfile.NamedTemporaryFile(delete=False) as tmpf:
            init_method = f"file://{tmpf.name}"
        mp.spawn(raw_broadcast_worker, args=(world_size, repeats, init_method, task_name, config_file), nprocs=world_size, join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1, help='number of iterations')
    parser.add_argument('--task-name', type=str, default="model_cfg", help='module name for dynamic task config')
    parser.add_argument('--config-file', type=str, default=None, help='path to task config file')
    args = parser.parse_args()
    benchmark(args.repeats, args.task_name, args.config_file)
