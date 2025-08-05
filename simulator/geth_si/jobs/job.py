"""
Job management and characterization module.
"""
from enum import Enum, auto
from typing import Dict, List, Optional
import attr

import os
import json
from typing import Any, Dict
from simpy import Event
from geth_si.cluster.resource import GPU

def load_all_results(results_root: str) -> Dict[str, Any]:
    """
    Walks plots/overall_results/results, reads each sub‐folder's result.json,
    and returns a dict mapping folder_name -> parsed JSON content.
    """
    all_results: Dict[str, Any] = {}
    if not os.path.isdir(results_root):
        raise ValueError(f"{results_root!r} is not a directory")

    for entry in os.listdir(results_root):
        folder_path = os.path.join(results_root, entry)
        json_path = os.path.join(folder_path, "results.json")
        if os.path.isdir(folder_path) and os.path.isfile(json_path):
            with open(json_path, "r") as f:
                try:
                    all_results[entry] = json.load(f)
                except json.JSONDecodeError as e:
                    # handle bad/malformed JSON if needed
                    print(f"Failed to decode {json_path}: {e}")
    return all_results

job_details = load_all_results("/workspace/results")

class JobStatus(Enum):
    """Possible states of a training job."""
    CREATED = auto()
    QUEUED = auto()
    ALLOCATED = auto()
    RUNNING = auto()
    PRE_SCALING = auto()
    SCALING = auto()
    COMPLETED = auto()
    FAILED = auto()

@attr.s(auto_attribs=True)
class JobRequirements:
    """Resource requirements for a job."""
    min_gpus: int
    max_gpus: int
    prefer_same_machine: bool = True

@attr.s(auto_attribs=True)
class JobMetrics:
    """Performance metrics for a job."""
    start_time: Optional[float] = None
    queued_time: Optional[float] = None
    end_time: Optional[float] = None
    performance_history: List[float] = attr.field(factory=list)
    # progress_history: List[float] = attr.field(factory=list)
    # Track GPU allocation history with timestamps
    gpu_allocation_history: List[Dict[str, any]] = attr.field(factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

@attr.s(auto_attribs=True)
class Job:
    """Represents a deep learning training job."""
    job_id: str
    name: str
    task_type: str
    requirements: JobRequirements
    target_epoch: int = 1
    samples_per_epoch: int = 100
    current_epoch: int = 0
    current_sample: int = 0
    batch_size: int = 4
    status: JobStatus = JobStatus.CREATED
    job_finished_event: Optional[Event] = None

    metrics: JobMetrics = attr.Factory(JobMetrics)
    assigned_gpus: list[GPU] = attr.field(factory=list)
    current_gpus: list[GPU] = attr.field(factory=list)
    pending_scale: Optional[Dict[str, any]] = None
    
    def __str__(self) -> str:
        lines = [
            f"Job ID: {self.job_id}",
            f"Name: {self.name}",
            f"Type: {self.task_type}",
            f"Min GPUs: {self.requirements.min_gpus}",
            f"Max GPUs: {self.requirements.max_gpus}",
            f"Target epochs: {self.target_epoch}",
            f"Total samples: {self.samples_per_epoch * self.target_epoch:}"
        ]
        return " | ".join(lines)
    
    def update_status(self, new_status: JobStatus, env_time: float) -> None:
        """
        Update job status and record metrics.
        
        Args:
            new_status: New status to set
            env_time: Current simulation environment time
        """
        old_status = self.status
        self.status = new_status
        
        if new_status == JobStatus.QUEUED and old_status != JobStatus.QUEUED:
            self.metrics.queued_time = env_time
        elif new_status == JobStatus.ALLOCATED and old_status != JobStatus.ALLOCATED:
            self.metrics.start_time = env_time
        elif new_status == JobStatus.RUNNING and old_status != JobStatus.RUNNING:
            self.metrics.running_time = env_time
        elif new_status in (JobStatus.COMPLETED, JobStatus.FAILED):
            self.metrics.end_time = env_time
    
    # def record_progress(self, progress: float) -> None:
    #     """Record a progress measurement."""
    #     self.metrics.progress_history.append(progress)

    def record_performance(self, performance_metric: float) -> None:
        """Record a performance measurement."""
        self.metrics.performance_history.append(performance_metric)
        
    def record_gpu_allocation(self, env_time: float) -> None:
        """Record current GPU allocation with timestamp."""
        # Update current_gpus list to match assigned_gpus
        self.current_gpus = self.assigned_gpus.copy()
        
        # Record allocation with timestamp
        self.metrics.gpu_allocation_history.append({
            "time": env_time,
            "gpu_count": len(self.current_gpus),
            "gpu_list": self.current_gpus.copy()
        })

    def get_job_speed(self, gpus=None) -> float:
        if gpus is None:
            gpus = self.current_gpus
            #wsq
            assert "gpus is None"
        job_type = self.task_type
        num_gpus = len(gpus)
        speed = 1 / job_details[job_type]["no_scale"][f"no_scale_{num_gpus}"]["avg_time"]
        return speed
    
    def get_remaining_time(self, gpu_num):
        remaining_samples = self.samples_per_epoch * (self.target_epoch - self.current_epoch) - self.current_sample
        gpu_list = [None] * gpu_num
        speed = self.get_job_speed(gpu_list)
        if self.samples_per_epoch == 1:
            remaining_time = remaining_samples / speed
        else:
            remaining_time = remaining_samples / (speed * self.batch_size * gpu_num)
        return remaining_time
    
    def check_scale_in_qualification(self, current_gpu_alloc, new_gpu_alloc):
        min_gpu_num = self.requirements.min_gpus
        remaining_samples = self.samples_per_epoch * (self.target_epoch - self.current_epoch) - self.current_sample
        current_speed = self.get_job_speed(current_gpu_alloc)
        overlap_time = self.get_scale_overlapped_time(cur_gpus=current_gpu_alloc, new_gpus=new_gpu_alloc)
        if self.samples_per_epoch == 1:
            remaining_samples_after_overlap = remaining_samples - overlap_time * current_speed
        else:
            remaining_samples_after_overlap = remaining_samples - overlap_time * current_speed * self.batch_size
        if remaining_samples_after_overlap <= 0:
            return -1
        new_speed = self.get_job_speed(new_gpu_alloc)
        if self.samples_per_epoch == 1:
            remaining_train_time = remaining_samples_after_overlap / new_speed
        else:
            remaining_train_time = remaining_samples_after_overlap / (new_speed * self.batch_size * len(new_gpu_alloc))
        scale_overhead_noncovered = self.get_scale_pause_time(cur_gpus=current_gpu_alloc, new_gpus=new_gpu_alloc)
        new_training_time = remaining_train_time + scale_overhead_noncovered
        # Create a dummy list with the correct number of GPUs for speed calculation
        min_gpu_list = [None] * min_gpu_num
        min_speed = self.get_job_speed(min_gpu_list)
        if self.samples_per_epoch == 1:
            max_remaining_time = remaining_samples_after_overlap / min_speed
        else:
            max_remaining_time = remaining_samples_after_overlap / (min_speed * self.batch_size * min_gpu_num)
        if new_training_time <= max_remaining_time:
            return True
        else:
            return False
    
    
    def get_benefit_or_disadvantage(self, current_gpu_alloc, new_gpu_alloc, benefit: bool = True):
        remaining_samples = self.samples_per_epoch * (self.target_epoch - self.current_epoch) - self.current_sample
        current_speed = self.get_job_speed(current_gpu_alloc)
        if self.samples_per_epoch == 1:
            current_est_fini_time = remaining_samples / current_speed
        else:
            current_est_fini_time = remaining_samples / (current_speed * self.batch_size * len(current_gpu_alloc))
        
        overlap_time = self.get_scale_overlapped_time(cur_gpus=current_gpu_alloc, new_gpus=new_gpu_alloc)
        if self.samples_per_epoch == 1:
            remaining_samples_after_overlap = remaining_samples - overlap_time * current_speed
        else:
            remaining_samples_after_overlap = remaining_samples - overlap_time * current_speed * self.batch_size
        if remaining_samples_after_overlap <= 0:
            return -1
        
        new_speed = self.get_job_speed(new_gpu_alloc)
        if self.samples_per_epoch == 1:
            remaining_train_time = remaining_samples_after_overlap / new_speed
        else:
            remaining_train_time = remaining_samples_after_overlap / (new_speed * self.batch_size * len(new_gpu_alloc))
        scale_overhead_noncovered = self.get_scale_pause_time(cur_gpus=current_gpu_alloc, new_gpus=new_gpu_alloc)
        difference = current_est_fini_time - overlap_time - scale_overhead_noncovered - remaining_train_time
        if benefit:
            return difference
        else:
            return -difference

    def get_env_setup_time(self) -> float:
        job_type = self.task_type
        num_gpus = len(self.assigned_gpus)
        env_setup_time = job_details[job_type]["no_scale"][f"no_scale_{num_gpus}"]["worker_init_time"]
        return env_setup_time
    
    def get_training_setup_time(self) -> float:
        job_type = self.task_type
        num_gpus = len(self.assigned_gpus)
        return job_details[job_type]["no_scale"][f"no_scale_{num_gpus}"]["training_setup_time"]
    
    def get_env_setup_time_with_partition(self) -> float:
        job_type = self.task_type
        num_gpus = len(self.assigned_gpus)
        return self.get_env_setup_time() + job_details[job_type]["no_scale"][f"no_scale_{num_gpus}"]["training_setup_time"]
    
    def get_scale_overlapped_time(self, cur_gpus=None, new_gpus=None) -> float:
        if os.environ.get("GETH_SI_SCALE_METHOD", "GETH") == "NAIVE":
            return 0

        if cur_gpus is None:
            cur_gpus = self.current_gpus
        if new_gpus is None:
            new_gpus = self.assigned_gpus
        job_type = self.task_type
        from_gpus = len(cur_gpus)
        to_gpus = len(new_gpus)
        scale_overlapped_time = 0
        scale_overlapped_time = job_details[job_type]["scale"][f"scale_{from_gpus}_{to_gpus}"]["overlapped_scale_time"]
        scale_overlapped_time = max(scale_overlapped_time, job_details[job_type]["scale"][f"scale_{from_gpus}_{to_gpus}"]["non_overlapped_scale_time"].get("prepartition_wait_time", 0))
        return scale_overlapped_time

    def get_scale_pause_time(self, cur_gpus=None, new_gpus=None) -> float:
        if cur_gpus is None:
            cur_gpus = self.current_gpus
        if new_gpus is None:
            new_gpus = self.assigned_gpus
        job_type = self.task_type
        from_gpus = len(cur_gpus)
        to_gpus = len(new_gpus)

        if os.environ.get("GETH_SI_SCALE_METHOD", "GETH") == "NAIVE":
            key = f"no_scale_no_micro_{to_gpus}"
            if key not in job_details[job_type]["no_scale"]:
                key = f"no_scale_{to_gpus}"
            return job_details[job_type]["no_scale"][key]["worker_init_time"] + job_details[job_type]["no_scale"][key]["training_setup_time"]

        scale_pause_time = job_details[job_type]["scale"][f"scale_{from_gpus}_{to_gpus}"]["non_overlapped_scale_time"]["total"] - job_details[job_type]["scale"][f"scale_{from_gpus}_{to_gpus}"]["non_overlapped_scale_time"].get("prepartition_wait_time", 0)
        return scale_pause_time

    def estimate_remaining_time(self):
        current_gpu_alloc = self.current_gpus
        remaining_samples = self.samples_per_epoch * (self.target_epoch - self.current_epoch) - self.current_sample
        current_speed = self.get_job_speed()
        if self.samples_per_epoch == 1:
            current_est_fini_time = remaining_samples / current_speed
        else:
            current_est_fini_time = remaining_samples / (current_speed * self.batch_size * len(current_gpu_alloc))
        return current_est_fini_time

    def get_remaining_percentage(self):
        remaining_samples = self.samples_per_epoch * (self.target_epoch - self.current_epoch) - self.current_sample
        total_samples = self.samples_per_epoch * self.target_epoch
        return remaining_samples / total_samples
