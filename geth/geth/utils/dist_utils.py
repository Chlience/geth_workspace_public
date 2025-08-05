import io
import pickle
from typing import Dict, List

import torch
import torch.distributed as dist


def wait_for_all_async_reqs(reqs: List[dist.Work]):
    for req in reqs:
        req.wait()


def _serialize_object(obj):
    # Serialize object to a byte tensor
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)
    byte_array = buffer.read()
    byte_tensor = torch.ByteTensor(list(byte_array))
    return byte_tensor


def _deserialize_object(byte_tensor):
    # Deserialize object from a byte tensor
    byte_array = byte_tensor.numpy().tobytes()
    buffer = io.BytesIO(byte_array)
    obj = pickle.load(buffer)
    return obj


def send_pyobject(obj, target_list: List[int]):
    obj_tensor = _serialize_object(obj)
    # send size tensor
    size_tensor = torch.tensor([obj_tensor.size(0)], dtype=torch.long)
    send_op_list = []
    for target in target_list:
        send_op = dist.P2POp(dist.isend, size_tensor, target)
        send_op_list.append(send_op)
    reqs = dist.batch_isend_irecv(send_op_list)
    wait_for_all_async_reqs(reqs)
    # send actual obj tensor
    send_op_list = []
    for target in target_list:
        send_op = dist.P2POp(dist.isend, obj_tensor, target)
        send_op_list.append(send_op)
    reqs = dist.batch_isend_irecv(send_op_list)
    wait_for_all_async_reqs(reqs)


def recv_pyobject(src: int):
    size_tensor = torch.tensor([0], dtype=torch.long)
    dist.recv(size_tensor, src)
    obj_tensor_size = size_tensor.item()
    assert isinstance(obj_tensor_size, int)
    obj_tensor = torch.empty(obj_tensor_size, dtype=torch.uint8)
    dist.recv(obj_tensor, src)
    obj = _deserialize_object(obj_tensor)
    return obj


def send_state_dict(state_dict: Dict, target_list: List[int]):
    keys = sorted(state_dict.keys())
    non_tensor_dict = {}
    send_op_list = []
    for k in keys:
        if not isinstance(state_dict[k], torch.Tensor):
            non_tensor_dict[k] = state_dict[k]
            continue
        for target in target_list:
            send_op = dist.P2POp(dist.isend, state_dict[k], target)
            send_op_list.append(send_op)

    if len(send_op_list) != 0:
        reqs = dist.batch_isend_irecv(send_op_list)
        wait_for_all_async_reqs(reqs)
    if len(non_tensor_dict) != 0:
        # send other data
        send_pyobject(non_tensor_dict, target_list)


def recv_state_dict(state_dict: Dict, src: int):
    keys = sorted(state_dict.keys())
    recv_op_list = []
    non_tensor_obj_cnt = 0
    for k in keys:
        if not isinstance(state_dict[k], torch.Tensor):
            non_tensor_obj_cnt += 1
            continue
        recv_op = dist.P2POp(dist.irecv, state_dict[k], src)
        recv_op_list.append(recv_op)

    if len(recv_op_list) != 0:
        reqs = dist.batch_isend_irecv(recv_op_list)
        wait_for_all_async_reqs(reqs)

    # recover other data
    if non_tensor_obj_cnt != 0:
        non_tensor_dict = recv_pyobject(src)
        for k, v in non_tensor_dict.items():
            state_dict[k] = v
