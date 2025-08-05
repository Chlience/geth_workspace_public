"""
Base scheduler implementation and interfaces.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Set
import attr
import simpy
from ..jobs.job import Job, JobStatus
from ..jobs.job_proc import JobProc
from ..cluster.resource import ClusterState, GPU
from ..jobs.job import JobMetrics

@attr.s(auto_attribs=True)
class SchedulerMetrics:
    """Metrics collected by the scheduler."""
    total_jobs: int = 0
    finished_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_queue_time: float = 0.0
    total_completion_time: float = 0.0
    detailed_job_metrics: Dict[str, JobMetrics] = attr.ib(factory=dict)

class Scheduler(ABC):
    """Abstract base class for job schedulers."""
    
    def __init__(self, env: simpy.Environment):
        self.env: simpy.Environment = env
        self.queue: List[Job] = []
        self.running_jobs: Dict[str, Job] = {}
        self.cluster: Optional[ClusterState] = None
        self.metrics = SchedulerMetrics()
        self.running_job_procs: Dict[str, JobProc] = {}
        self.all_jobs: List[Job] = []

    def set_cluster(self, cluster: ClusterState) -> None:
        """Set the cluster state for this scheduler."""
        self.cluster = cluster

    @abstractmethod
    def schedule(self) -> None:
        """Implement scheduling logic in derived classes."""
        pass

    def submit_job(self, job: Job) -> None:
        """Submit a new job to the scheduler."""
        job.update_status(JobStatus.QUEUED, self.env.now)
        job.job_finished_event = self.env.event()
        self.queue.append(job)
        self.all_jobs.append(job)
        self.metrics.total_jobs += 1

    def complete_job(self, job: Job, success: bool = True) -> None:
        """Mark a job as completed."""
        if success:
            job.update_status(JobStatus.COMPLETED, self.env.now)
            self.metrics.completed_jobs += 1
        else:
            job.update_status(JobStatus.FAILED, self.env.now)
            self.metrics.failed_jobs += 1
        job.job_finished_event.succeed()
        
        if job.job_id in self.running_jobs:
            del self.running_jobs[job.job_id]
            del self.running_job_procs[job.job_id]
            self.metrics.total_completion_time += job.metrics.end_time - job.metrics.queued_time
            self.metrics.total_queue_time += job.metrics.start_time - job.metrics.queued_time
            self.metrics.finished_jobs += 1
            self.metrics.detailed_job_metrics[job.job_id] = job.metrics
            
        # Release GPUs
        if self.cluster:
            self.cluster.release_gpus(job.job_id)

    def get_metrics(self) -> SchedulerMetrics:
        """Get current scheduler metrics."""
        return self.metrics

    def get_queue_task_info(self) -> Dict[str, Dict]:
        """Get detailed information about the current tasks."""
        task_info = {}
        for job in self.queue:
            task_info[job.job_id] = {
                'task_type': job.task_type,
                'submit_time': job.metrics.queued_time,
                'min_gpu': job.requirements.min_gpus,
            }
        return task_info

class FIFOScheduler(Scheduler):
    """Simple FIFO (First-In-First-Out) scheduler implementation."""
    
    def schedule(self) -> None:
        """Implement FIFO scheduling logic."""
        if not self.cluster:
            return

        # Process queue in FIFO order
        remaining_queue = []
        should_alloc = True
        strict_fifo = True
        for job in self.queue:
            if strict_fifo and not should_alloc:
                remaining_queue.append(job)
                continue

            # Try to allocate GPUs
            allocated_gpus = self.cluster.pre_allocate_gpus(
                job_id=job.job_id,
                num_gpus=job.requirements.min_gpus,
                prefer_same_machine=job.requirements.prefer_same_machine
            )
            
            if allocated_gpus:
                # Successfully allocated GPUs
                job.assigned_gpus = [gpu for gpu in allocated_gpus]
                job.update_status(JobStatus.ALLOCATED, self.env.now)
                self.running_jobs[job.job_id] = job
                self.running_job_procs[job.job_id] = JobProc(self.env, job, self)
            else:
                # Couldn't allocate enough GPUs
                should_alloc = False
                remaining_queue.append(job)
        
        self.queue = remaining_queue


class SJFScheduler(Scheduler):
    """Simple SJF (Shortest Job First) scheduler implementation."""
    
    def schedule(self) -> None:
        """Implement SJF scheduling logic."""
        if not self.cluster:
            return

        # Process queue in FIFO order
        remaining_queue = []
        should_alloc = True
        strict_fifo = False
        self.queue.sort(key=lambda j: j.get_remaining_time(j.requirements.min_gpus))
        for job in self.queue:
            if strict_fifo and not should_alloc:
                remaining_queue.append(job)
                continue

            # Try to allocate GPUs
            allocated_gpus = self.cluster.pre_allocate_gpus(
                job_id=job.job_id,
                num_gpus=job.requirements.min_gpus,
                prefer_same_machine=job.requirements.prefer_same_machine
            )
            
            if allocated_gpus:
                # Successfully allocated GPUs
                job.assigned_gpus = [gpu for gpu in allocated_gpus]
                job.update_status(JobStatus.ALLOCATED, self.env.now)
                self.running_jobs[job.job_id] = job
                self.running_job_procs[job.job_id] = JobProc(self.env, job, self)
            else:
                # Couldn't allocate enough GPUs
                should_alloc = False
                remaining_queue.append(job)
        
        self.queue = remaining_queue

# Import the ElasticScheduler from the elastic module
from .elastic import ElasticScheduler
from .elastic_with_prestart import ElasticPrestartScheduler
from .elastic_with_prestart_sjf import ElasticPrestartSjfScheduler