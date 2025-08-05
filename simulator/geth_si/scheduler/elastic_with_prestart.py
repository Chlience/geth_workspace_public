"""
Elastic scheduler implementation.
"""
from typing import Set, Dict, List, Tuple
from loguru import logger
from geth_si.jobs.job_proc import JobProc
import simpy
from .base import Scheduler
from ..jobs.job import Job, JobStatus
from ..cluster.resource import ClusterState, GPU, get_gpu_ids

STOP_SCALE_LIMIT = 10
PRE_START_LIMIT = STOP_SCALE_LIMIT + 5
BEFORE_SCALE_LIMIT = 20

class ElasticPrestartScheduler(Scheduler):
    def __init__(self, env: simpy.Environment):
        super().__init__(env)
        self.scaled_jobs: Set[str] = set()  # Track jobs that have been scaled
        self.pending_gpu_allocations = {}  # Track pending GPU allocations for each job
    
    
    def schedule(self) -> None:
        """Implement elastic scheduling logic."""
        if not self.cluster:
            return
        
        # schedule by machine and job
        scheduled_jobs = set()
        
        # wsq
        scale_out_jobs = []
        for job_id, job in list(self.running_jobs.items()):
            current_gpus = len(job.assigned_gpus)
            if len(job.current_gpus) != 0 and job.estimate_remaining_time() < PRE_START_LIMIT:
                continue
            if job.status != JobStatus.RUNNING:
                continue
            if self.env.now - job.metrics.running_time < BEFORE_SCALE_LIMIT:
                continue
            if current_gpus < job.requirements.max_gpus:
                # This job can potentially be scaled out
                scale_out_jobs.append(job)
                    
        scale_in_jobs = []
        for job_id, job in list(self.running_jobs.items()):
            current_gpus = len(job.assigned_gpus)
            if len(job.current_gpus) != 0 and job.estimate_remaining_time() < PRE_START_LIMIT:
                continue
            if current_gpus > 1:
                # This job can potentially be scaled in
                scale_in_jobs.append(job)
        
        jobs_to_scale = set()
        
        for machine_id, machine in self.cluster.machines.items():
            # for each machine, check number of available gpus within some time, not including preallocated gpus
            machine = self.cluster.get_machine(machine_id)
            available_gpus = []
            for gpu_id, gpu in machine.gpus.items():
                if gpu.pre_allocated:
                    continue
                if gpu.available:
                    available_gpus.append(gpu)
                    continue
                job = self.running_jobs[gpu.current_job]
                if job.current_gpus != job.assigned_gpus:
                    continue
                est_fini_time = job.estimate_remaining_time()
                if est_fini_time < PRE_START_LIMIT:
                    available_gpus.append(gpu)
            
            # 尝试调度新任务
            new_job_flag = False
            for job in self.queue:
                if job.job_id in scheduled_jobs:
                    continue
                if len(available_gpus) >= job.requirements.min_gpus:
                    choosed_gpus = available_gpus[:job.requirements.min_gpus]
                    available_gpus = available_gpus[job.requirements.min_gpus:]
                    job.assigned_gpus = []
                    for gpu in choosed_gpus:
                        gpu.pre_allocate(job.job_id)
                        job.assigned_gpus.append(gpu)
                    self.running_jobs[job.job_id] = job
                    self.running_job_procs[job.job_id] = JobProc(self.env, job, self)
                    scheduled_jobs.add(job.job_id)
                    new_job_flag = True
            self.queue = [j for j in self.queue if j.job_id not in scheduled_jobs]
            scale_in_flag = False
            # wsq 如果1个machine里面有空gpu却没有新任务就走我们的新的 如果有新任务上去了就走原来的
            # 尝试 scale in
            if not new_job_flag and len(available_gpus) > 0 and self.queue:
            # if not new_job_flag and self.queue:
                job_disadvantages = []
                # wsq calculate the number of gpus needed for the new task
                min_requirement = 100
                for job in self.queue:
                    if job.requirements.min_gpus < min_requirement:
                        min_requirement = job.requirements.min_gpus
                        pending_job = job
                needed_gpus = pending_job.requirements.min_gpus-len(available_gpus)
                if needed_gpus <= 0:
                    assert False, "pending job has enough gpus"

                for job in scale_in_jobs:
                    if len(job.current_gpus) <= needed_gpus:
                        continue
                    current_machine = job.current_gpus[0].machine_id
                    if current_machine != machine_id:
                        continue
                    if len(job.current_gpus) != len(job.assigned_gpus):
                        continue # already scaling
                    current_allocation = job.current_gpus
                    new_allocation = current_allocation[:-needed_gpus]
                    if not job.check_scale_in_qualification(current_allocation, new_allocation):
                        continue
                    # wsq calculate the marginal disadvantage using class method
                    marginal_disadvantage = job.get_benefit_or_disadvantage(current_allocation, new_allocation, benefit=False)
                    job_disadvantages.append((job, marginal_disadvantage))
                if len(job_disadvantages) > 0:
                    job_disadvantages.sort(key=lambda x: x[1], reverse=True)
                    best_job, _ = job_disadvantages[0]
                    # wsq Allocate the GPU to pending job
                    for gpu in best_job.current_gpus[-needed_gpus:]:
                        gpu.pre_allocate(pending_job.job_id)
                        pending_job.assigned_gpus.append(gpu)
                        best_job.assigned_gpus.remove(gpu)
                        
                    logger.debug(f"[{self.env.now:.2f}] Scale in {best_job.job_id} from {len(best_job.current_gpus)} to {len(best_job.assigned_gpus)}")
                    logger.debug(f"wsq schedule {pending_job.job_id}")
                    self.running_jobs[pending_job.job_id] = pending_job
                    self.running_job_procs[pending_job.job_id] = JobProc(self.env, pending_job, self)
                    scheduled_jobs.add(pending_job.job_id)
                    self.queue = [j for j in self.queue if j.job_id not in scheduled_jobs]
                    scale_in_flag = True      
                    # Mark job as scaled
                    self.scaled_jobs.add(best_job.job_id)
                    jobs_to_scale.add(best_job.job_id)
                    
                    # Check if job reached max capacity
                    # if len(best_job.assigned_gpus) <= best_job.requirements.min_gpus:
                    scale_in_jobs.remove(best_job)
            # 如果没有 scale in，并且还有空余 gpu，考虑 scale out
            if len(available_gpus) > 0 and not scale_in_flag:
                # Calculate marginal benefit for each job and allocate GPUs based on that
                
                # Calculate marginal benefit for each job
                job_benefits = []

                for job in scale_out_jobs:
                    if len(job.current_gpus) == 0:
                        continue # todo: fixme
                    
                    current_machine = job.current_gpus[0].machine_id
                    if current_machine != machine_id:
                        continue

                    # 有问题 这样会把初始的job看作不能扩的，但实际上应该看看初始的job是否能扩，再按真实gpu数目启动它
                    if len(job.current_gpus) != len(job.assigned_gpus):
                        continue # already scaling

                    available_gpus = []
                    for gpu_id, gpu in machine.gpus.items():
                        if gpu.pre_allocated:
                            continue
                        if gpu.available:
                            available_gpus.append(gpu)
                            continue
                        if gpu.current_job == job.job_id:
                            continue
                        gpu_job = self.running_jobs[gpu.current_job]
                        if gpu_job.current_gpus != gpu_job.assigned_gpus:
                            continue
                        est_fini_time = gpu_job.estimate_remaining_time()
                        # this job cannot be scaling since est_fini_time < STOP_SCALE_LIMIT
                        if est_fini_time < job.get_env_setup_time() and est_fini_time < STOP_SCALE_LIMIT:
                            available_gpus.append(gpu)
                    if len(available_gpus) == 0:
                        continue

                    for i in range(len(available_gpus)):
                        # Calculate current job speed
                        current_gpus = len(job.current_gpus)
                        
                        current_allocation = job.current_gpus

                        potential_allocation = available_gpus[:i+1]
                        new_allocation = current_allocation.copy()
                        new_allocation.extend(potential_allocation)
                        
                        # Calculate marginal benefit using class method
                        marginal_benefit = job.get_benefit_or_disadvantage(current_allocation, new_allocation, benefit=True)

                        logger.debug(f"{job.job_id} scale out from {len(current_allocation)} to {len(new_allocation)} marginal benefit {marginal_benefit:.3f}")
                        if marginal_benefit > 0:
                            job_benefits.append((job, potential_allocation, marginal_benefit))
                
                # If no jobs benefit from more GPUs, break
                if not job_benefits:
                    continue
                    
                # Sort by marginal benefit (highest first)
                job_benefits.sort(key=lambda x: x[2], reverse=True)
                if job_benefits[0][2] < 0:
                    continue
                
                # Allocate GPU to the job with highest marginal benefit
                best_job, best_gpu, _ = job_benefits[0]
                
                # Allocate the GPU
                for g in best_gpu:
                    g.pre_allocate(best_job.job_id)
                    best_job.assigned_gpus.append(g)

                logger.debug(f"[{self.env.now:.2f}]: Job {best_job.job_id} scale out from {len(best_job.current_gpus)} to {len(best_job.assigned_gpus)} marginal benefit {job_benefits[0][2]}")
                
                # Mark job as scaled
                self.scaled_jobs.add(best_job.job_id)
                jobs_to_scale.add(best_job.job_id)
                
                # Check if job reached max capacity
                if len(best_job.assigned_gpus) >= best_job.requirements.max_gpus:
                    scale_out_jobs.remove(best_job)
        
        self.queue = [j for j in self.queue if j.job_id not in scheduled_jobs]
        for job_id in jobs_to_scale:
            self.running_job_procs[job_id].trigger_scale()
    
    def complete_job(self, job: Job, success: bool = True) -> None:
        """Mark a job as completed and remove from scaled jobs set if present."""
        super().complete_job(job, success)
        
        # Remove from scaled jobs set if present
        if job.job_id in self.scaled_jobs:
            self.scaled_jobs.remove(job.job_id)
