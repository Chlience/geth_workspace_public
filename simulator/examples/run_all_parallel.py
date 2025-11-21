#!/usr/bin/env python3
import argparse
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def run_simulation(scheduler: str = "elastic_pre_sjf", elastic_type: str = "geth", 
                   seed: int = 0, task_num: int = None, log_dir: str = "logs", 
                   phy_mode: bool = False):
    """Run a single simulation and return the result."""
    if task_num is not None:
        log_file = Path(log_dir) / f"{scheduler}_simulation_seed_{seed}_task_{task_num}.log"
    else:
        log_file = Path(log_dir) / f"{scheduler}_simulation_seed_{seed}.log"
    
    cmd = [
        "python3", 
        "examples/simulation_from_list.py",
        f"--scheduler={scheduler}",
        f"--elastic_type={elastic_type}",
        f"--seed={seed}"
    ]
    
    if task_num is not None:
        cmd.append(f"--task_num={task_num}")
    if phy_mode:
        cmd.append("--phy")
    
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        return (scheduler, elastic_type, seed, task_num, "SUCCESS")
    except subprocess.CalledProcessError as e:
        return (scheduler, elastic_type, seed, task_num, f"FAILED: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run blockchain simulations with different schedulers")
    parser.add_argument("--seed_begin", type=int, default=0, help="Starting seed value")
    parser.add_argument("--seed_end", type=int, default=100, help="Ending seed value")
    parser.add_argument("--task_num_begin", type=int, default=None, help="Starting number of tasks to simulate")
    parser.add_argument("--task_num_end", type=int, default=None, help="Ending number of tasks to simulate")
    parser.add_argument("--log_dir", default="logs", help="Directory to store log files")
    parser.add_argument("--phy", action='store_true', help="Run in physical mode")
    parser.add_argument("--max-workers", type=int, default=None, 
                       help="Maximum number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    
    configurations = [
        ("elastic_pre_sjf", "geth"),
        ("sjf", "naive"),
        ("fifo", "naive"),
    ]
    
    task_num_list = [20]
    
    if args.task_num_begin is not None and args.task_num_end is not None:
        task_num_list = list(range(args.task_num_begin, args.task_num_end + 1, 10))
    
    # Track execution time
    start_time = time.time()
    total_tasks = (args.seed_end - args.seed_begin + 1) * len(task_num_list) * len(configurations)
    completed_tasks = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        
        # Submit all tasks to the executor
        for seed in range(args.seed_begin, args.seed_end + 1):
            for task_num in task_num_list:
                for scheduler, elastic_type in configurations:
                    future = executor.submit(
                        run_simulation,
                        scheduler=scheduler,
                        elastic_type=elastic_type,
                        seed=seed,
                        task_num=task_num,
                        log_dir=args.log_dir,
                        phy_mode=args.phy
                    )
                    futures.append(future)
        
        # Process results as they complete
        for future in as_completed(futures):
            scheduler, elastic_type, seed, task_num, status = future.result()
            completed_tasks += 1
            print(f"[{completed_tasks}/{total_tasks}] {status}: {scheduler}, {elastic_type}, seed={seed}, task_num={task_num}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nAll simulations completed in {elapsed_time:.2f} seconds")
    print(f"Total tasks: {total_tasks}")
    print(f"Logs saved in: {args.log_dir}")

if __name__ == "__main__":
    main()