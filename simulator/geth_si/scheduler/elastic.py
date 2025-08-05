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

class ElasticScheduler(Scheduler):
    """
    Elastic scheduler that can scale jobs when more GPUs become available.
    This scheduler prioritizes:
    1. Starting new jobs with minimum GPU requirements
    2. Scaling up running jobs to their maximum GPU requirements
    """
    
    def __init__(self, env: simpy.Environment):
        super().__init__(env)
        self.scaled_jobs: Set[str] = set()  # Track jobs that have been scaled
        self.pending_gpu_allocations = {}  # Track pending GPU allocations for each job
    
    def schedule(self) -> None:
        """Implement elastic scheduling logic."""
        if not self.cluster:
            return
            
        # First, try to start new jobs with empty gpus
        remaining_queue = []
        for job in self.queue:
            allocated_gpus = self.cluster.pre_allocate_gpus(
                job_id=job.job_id,
                num_gpus=job.requirements.min_gpus,
                prefer_same_machine=job.requirements.prefer_same_machine
            )
            
            if allocated_gpus:
                # Successfully allocated GPUs
                job.assigned_gpus = [gpu for gpu in allocated_gpus]
                self.running_jobs[job.job_id] = job
                self.running_job_procs[job.job_id] = JobProc(self.env, job, self)
            else:
                # Couldn't allocate enough GPUs
                remaining_queue.append(job)
        
        self.queue = remaining_queue
        
        # Calculate marginal benefit for each job and allocate GPUs based on that
        scalable_jobs = []
        for job_id, job in list(self.running_jobs.items()):
            current_gpus = len(job.assigned_gpus)
            if current_gpus < job.requirements.max_gpus:
                # This job can potentially be scaled up
                scalable_jobs.append(job)
        
        
        jobs_to_scale = set()
        available_gpus_by_machine = self.cluster.get_all_available_gpus()
        for machine_id, available_gpus in available_gpus_by_machine.items():
            if len(available_gpus) == 0:
                continue

            # Calculate marginal benefit for each job
            job_benefits = []
            for job in scalable_jobs:
                if len(job.current_gpus) == 0:
                    continue # todo: fixme
                
                current_machine = job.current_gpus[0].machine_id
                if current_machine != machine_id:
                    continue

                if len(job.current_gpus) != len(job.assigned_gpus):
                    continue # already scaling

                for i in range(len(available_gpus)):
                    # Calculate current job speed
                    current_gpus = len(job.current_gpus)
                    
                    current_allocation = job.current_gpus

                    potential_allocation = available_gpus[:i+1]
                    new_allocation = current_allocation.copy()
                    new_allocation.extend(potential_allocation)
                    
                    # Calculate marginal benefit using class method
                    marginal_benefit = job.get_benefit_or_disadvantage(current_allocation, new_allocation, benefit=True)

                    # logger.debug(f"{job.job_id} scale from {get_gpu_ids(current_allocation)} to {get_gpu_ids(new_allocation)} marginal benefit {marginal_benefit:.3f}")
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
            
            # Mark job as scaled
            self.scaled_jobs.add(best_job.job_id)
            jobs_to_scale.add(best_job.job_id)
            
            # Check if job reached max capacity
            if len(best_job.assigned_gpus) >= best_job.requirements.max_gpus:
                scalable_jobs.remove(best_job)
        
        for job_id in jobs_to_scale:
            self.running_job_procs[job_id].trigger_scale()
    
    def complete_job(self, job: Job, success: bool = True) -> None:
        """Mark a job as completed and remove from scaled jobs set if present."""
        super().complete_job(job, success)
        
        # Remove from scaled jobs set if present
        if job.job_id in self.scaled_jobs:
            self.scaled_jobs.remove(job.job_id)
