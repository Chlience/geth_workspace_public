"""
Elastic scheduler implementation.
"""
from typing import Set, Dict, List, Tuple
from geth_si.jobs.job_proc import JobProc
import simpy
from .base import Scheduler
from ..jobs.job import Job, JobStatus
from ..cluster.resource import ClusterState, GPU

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
            
        # First, try to start new jobs with minimum requirements
        remaining_queue = []
        for job in self.queue:
            # Try to allocate minimum GPUs
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
        # First, collect all jobs that can be scaled
        scalable_jobs = []
        for job_id, job in list(self.running_jobs.items()):
            current_gpus = len(job.assigned_gpus)
            if current_gpus < job.requirements.max_gpus:
                # This job can potentially be scaled up
                scalable_jobs.append(job)
        
        # Get available GPUs from the cluster
        available_gpus_by_machine = self.cluster.get_all_available_gpus()
        available_gpus = [gpu for gpus in available_gpus_by_machine.values() for gpu in gpus]
        
        # Continue allocating GPUs until no more are available or no jobs benefit from more GPUs
        while available_gpus and scalable_jobs:
            # Calculate marginal benefit for each job
            job_benefits = []
            for job in scalable_jobs:
                # Calculate current job speed
                current_gpus = len(job.assigned_gpus)
                current_allocation = job.assigned_gpus

                # Consider topology: same machine if possible
                potential_allocation = available_gpus[0]
                current_machines = set(map(lambda gpu: gpu.machine_id, current_allocation))
                if potential_allocation.machine_id not in current_machines:
                    continue

                new_allocation = current_allocation.copy()
                new_allocation.append(potential_allocation)
                
                # Calculate marginal benefit using class method
                marginal_benefit = job.get_benefit(current_allocation, new_allocation)
                
                job_benefits.append((job, potential_allocation, marginal_benefit))
            
            # If no jobs benefit from more GPUs, break
            if not job_benefits:
                break
                
            # Sort by marginal benefit (highest first)
            job_benefits.sort(key=lambda x: x[2], reverse=True)
            if job_benefits[0][2] < 0:
                break
            
            # Allocate GPU to the job with highest marginal benefit
            best_job, best_gpu, _ = job_benefits[0]
            
            # Allocate the GPU
            best_gpu.pre_allocate(best_job.job_id)
            best_job.assigned_gpus.append(best_gpu)
            
            # Mark job as scaled
            self.scaled_jobs.add(best_job.job_id)
            
            # Remove the allocated GPU from available GPUs
            available_gpus.remove(best_gpu)
            
            # Check if job reached max capacity
            if len(best_job.assigned_gpus) >= best_job.requirements.max_gpus:
                scalable_jobs.remove(best_job)
    
    def _calculate_marginal_benefit(self, job: Job, current_allocation: Dict[str, int], new_allocation: Dict[str, int]) -> float:
        """Calculate the marginal benefit of adding a GPU to a job."""
        pass
        # return new_speed - current_speed + job.get_scale_overhead(len(current_allocation), len(new_allocation))
        return -1
    
    def complete_job(self, job: Job, success: bool = True) -> None:
        """Mark a job as completed and remove from scaled jobs set if present."""
        super().complete_job(job, success)
        
        # Remove from scaled jobs set if present
        if job.job_id in self.scaled_jobs:
            self.scaled_jobs.remove(job.job_id)
    
    def _get_best_gpu_to_add(self, job: Job, available_gpus: List[GPU]):
        """Find the best GPU to add to a job considering topology."""
        if not available_gpus:
            return None
            
        # Get current allocation
        current_allocation = self._get_gpu_allocation_for_job(job)
        current_machines = set(current_allocation.keys())
        
        # If job prefers same machine, try to find a GPU on a machine it's already using
        if job.requirements.prefer_same_machine and current_machines:
            # Find GPUs on machines the job is already using
            same_machine_gpus = [gpu for gpu in available_gpus if gpu.machine_id in current_machines]
            if same_machine_gpus:
                # Find the machine with the most GPUs already allocated to this job
                best_machine = max(current_machines, key=lambda m: current_allocation.get(m, 0))
                best_machine_gpus = [gpu for gpu in same_machine_gpus if gpu.machine_id == best_machine]
                if best_machine_gpus:
                    return best_machine_gpus[0]
                return same_machine_gpus[0]
        
        # If we can't find a GPU on the same machine or if the job doesn't prefer same machine
        # Try to find a GPU on a machine with the most available GPUs (for future scaling)
        machine_available_counts = {}
        for gpu in available_gpus:
            machine_available_counts[gpu.machine_id] = machine_available_counts.get(gpu.machine_id, 0) + 1
        
        # Find the machine with the most available GPUs
        if machine_available_counts:
            best_machine = max(machine_available_counts.keys(), key=lambda m: machine_available_counts[m])
            best_machine_gpus = [gpu for gpu in available_gpus if gpu.machine_id == best_machine]
            return best_machine_gpus[0]
        
        # If all else fails, just return the first available GPU
        return available_gpus[0]
