import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geth_si.jobs.job import Job, JobRequirements
from geth_si.simulation.engine import SimulationEngine, SimulationConfig

from random_trace_generator import generate_trace_with_timestamps

def run_simulation(seed: int, scheduler_type: str, enable_visualization=False, mode=None, task_num=None):
    # Create simulation configuration
    config = SimulationConfig(
        seed=seed,
        simulation_duration=3600 * 24,  # 1 hour
        scheduler_interval=1.0,
        enable_visualization=enable_visualization
    )
    
    # Initialize simulation engine
    engine = SimulationEngine(config, scheduler_type=scheduler_type)
    
    # Create cluster with 8 machines, each having 8 GPUs
    if mode == "phy":
        engine.add_machine("machine_0", num_gpus=8)
    else:
        engine.add_machine("machine_0", num_gpus=8)
        engine.add_machine("machine_1", num_gpus=8)
        engine.add_machine("machine_2", num_gpus=8)
        engine.add_machine("machine_3", num_gpus=8)
        engine.add_machine("machine_4", num_gpus=8)
        engine.add_machine("machine_5", num_gpus=8)
        engine.add_machine("machine_6", num_gpus=8)
        engine.add_machine("machine_7", num_gpus=8)
    
    if mode == "phy":
        job_num = 20
        scale = 0.6
    else:
        job_num = 260
        scale = 1
    
    if task_num is not None:
        job_num = task_num
    
    trace = generate_trace_with_timestamps(job_num, seed=seed, scale=scale)
    
    for idx, task in enumerate(trace):
        job = Job(
            job_id=f"job_{idx}",
            name=f"job_{idx}",
            task_type=task.task_type,
            requirements=JobRequirements(
                min_gpus=task.init_gpus,
                max_gpus=8,
            ),
            samples_per_epoch=task.samples,
            batch_size=task.batch_size,
            target_epoch=task.epochs,
        )
        engine.submit_job_at_time(job, task.arrive_time)
    
    # Run simulation
    metrics = engine.run()

    # Print results
    print(f"Seed {seed} {scheduler_type} Simulation Results:")
    print(f"Total Runtime: {metrics.total_runtime:.2f} seconds")
    # print(f"Avg Queue Time: {metrics.scheduler_matrics.total_queue_time / metrics.scheduler_matrics.completed_jobs:.2f} seconds")
    print(f"Avg Completion Time: {metrics.scheduler_matrics.total_completion_time / metrics.scheduler_matrics.completed_jobs:.2f} seconds")
    print(f"Completed Jobs: {metrics.scheduler_matrics.completed_jobs}\n")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with configurable seed and scheduler.")
    parser.add_argument('--seed', type=int, default=23, help='Random seed for simulation and trace generation (default: 42)')
    parser.add_argument('--scheduler', type=str, default='elastic_pre_sjf', help='Scheduler type for the simulation engine (default: elastic_pre_sjf)')
    parser.add_argument('--elastic_type', type=str, default='geth', help='Elastic type for the simulation engine (default: geth)')
    parser.add_argument('--log_dir', default='None', help='save log to file')
    parser.add_argument('--phy', action='store_true', help='physical mode')
    parser.add_argument('--task_num', type=int, default=None, help='Number of tasks to simulate (default: None, which means all tasks in the trace will be used)')
    args = parser.parse_args()

    if args.elastic_type == 'geth':
        os.environ['GETH_SI_SCALE_METHOD'] = 'GETH'
    elif args.elastic_type == 'naive':
        os.environ['GETH_SI_SCALE_METHOD'] = 'NAIVE'
    else:
        raise ValueError(f"Invalid elastic type: {args.elastic_type}")
    print(args.elastic_type, os.environ['GETH_SI_SCALE_METHOD'])
    
    if args.log_dir != 'None':
        from loguru import logger
        logger.remove()
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{current_time}_{args.seed}_{args.scheduler}_{args.elastic_type}.log"
        
        os.makedirs(args.log_dir, exist_ok=True)
        log_filepath = os.path.join(args.log_dir, log_filename)
        
        # 将日志输出到文件，并设置日志级别为 WARNING
        logger.add(
            sink=log_filepath,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        print(f"Logging to {log_filepath}")
        
    if args.phy:
        mode = "phy"
    else:
        mode = None

    run_simulation(seed=args.seed, scheduler_type=args.scheduler, mode=mode, task_num=args.task_num)
