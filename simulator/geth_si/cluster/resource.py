"""
Resource management module for cluster components with machine and GPU hierarchy.
"""
from typing import Dict, List, Optional, Set
import attr

from simpy import Event

def get_gpu_ids(gpus):
    """
    Extracts GPU IDs from a list of GPU objects.
    
    Args:
        gpus (list): List of GPU objects.
        
    Returns:
        list: List of GPU IDs.
    """
    gpu_ids = []
    for g in gpus:
        parts = g.gpu_id.split('_')
        gpu_ids.append((int(parts[1]), int(parts[3])))
    return gpu_ids

@attr.s(auto_attribs=True)
class GPU:
    """Represents a GPU in a machine."""
    gpu_id: str
    machine_id: str
    available: bool = True
    current_job: Optional[str] = None
    pre_allocated: bool = False
    pre_allocated_job: Optional[str] = None

    def allocate(self, job_id: str) -> bool:
        """Allocate this GPU to a job."""
        if not self.available:
            return False
        self.available = False
        self.current_job = job_id
        return True
    
    def pre_allocate(self, job_id: str) -> bool:
        """Pre-allocate this GPU to a job."""
        if self.pre_allocated:
            return False
        self.pre_allocated = True
        self.pre_allocated_job = job_id
        return True
    
    def commit_pre_alloc(self) -> bool:
        if not self.pre_allocated:
            return False
        if not self.available:
            return False
        self.available = False
        self.current_job = self.pre_allocated_job
        self.pre_allocated = False
        self.pre_allocated_job = None
        return True

    def release(self) -> None:
        """Release this GPU back to the pool."""
        self.available = True
        self.current_job = None

@attr.s(auto_attribs=True)
class Machine:
    """Represents a physical machine in the cluster."""
    machine_id: str
    gpus: Dict[str, GPU] = attr.Factory(dict)
    
    def add_gpu(self, gpu_id: str) -> None:
        """Add a GPU to this machine."""
        self.gpus[gpu_id] = GPU(gpu_id=gpu_id, machine_id=self.machine_id)
    
    def get_available_gpus(self) -> List[GPU]:
        """Get list of available GPUs in this machine."""
        return [gpu for gpu in self.gpus.values() if (gpu.available and not gpu.pre_allocated)]
    
    def get_gpu_by_id(self, gpu_id: str) -> Optional[GPU]:
        """Get a specific GPU by ID."""
        return self.gpus.get(gpu_id)

@attr.s(auto_attribs=True)
class ClusterState:
    """Manages the state of machines and GPUs in the cluster."""
    machines: Dict[str, Machine] = attr.Factory(dict)
    all_gpus: Dict[str, GPU] = attr.Factory(dict)

    def add_machine(self, machine_id: str, num_gpus: int) -> None:
        """
        Add a new machine with specified number of GPUs.

        Args:
            machine_id: Unique identifier for the machine.
            num_gpus: Number of GPUs to allocate to the machine.
        """
        machine = Machine(machine_id=machine_id)
        for i in range(num_gpus):
            gpu_id = f"{machine_id}_gpu_{i}"
            machine.add_gpu(gpu_id)
            self.all_gpus[gpu_id] = machine.gpus[gpu_id]
        self.machines[machine_id] = machine

    def get_machine(self, machine_id: str) -> Optional[Machine]:
        """Get a specific machine by ID."""
        return self.machines.get(machine_id)

    def get_gpu_by_id(self, gpu_id: str) -> Optional[GPU]:
        """Get a specific GPU by ID."""
        return self.all_gpus.get(gpu_id)

    def get_all_available_gpus(self) -> Dict[str, List[GPU]]:
        """Get all available GPUs grouped by machine."""
        available_gpus = {}
        for machine in self.machines.values():
            available = machine.get_available_gpus()
            if available:
                available_gpus[machine.machine_id] = available
        return available_gpus
    

    def pre_allocate_gpus(self, job_id: str, num_gpus: int, prefer_same_machine: bool = True, allow_partial: bool = False) -> Optional[List[GPU]]:
        """
        Allocate specified number of GPUs to a job.
        If prefer_same_machine is True, try to allocate from the same machine first.
        If allow_partial is True, will allocate as many GPUs as available up to num_gpus.
        Returns list of allocated GPUs if successful, None if failed or if no GPUs are available.
        """
        available_gpus = self.get_all_available_gpus()
        
        assert prefer_same_machine
        # Try to allocate from a single machine first
        for machine_id, gpus in available_gpus.items():
            if len(gpus) >= num_gpus:
                allocated_gpus = gpus[:num_gpus]
                for gpu in allocated_gpus:
                    gpu.pre_allocate(job_id)
                return allocated_gpus
        return None

    def release_gpus(self, job_id: str) -> None:
        """Release all GPUs allocated to a specific job."""
        for machine in self.machines.values():
            for gpu in machine.gpus.values():
                if gpu.current_job == job_id:
                    gpu.release()

    def get_gpu_allocation_info(self, gpus: List[GPU]) -> Dict[str, int]:
        """
        Get information about GPU allocation across machines.
        Returns a dictionary with machine IDs and number of GPUs allocated on each.
        """
        allocation = {}
        for gpu in gpus:
            allocation[gpu.machine_id] = allocation.get(gpu.machine_id, 0) + 1
        return allocation

    def get_cluster_info(self):
        """
        Get a summary of the cluster state.
        Returns a dictionary with machine IDs and their GPU allocation status.
        """
        info = {}
        for machine_id, machine in self.machines.items():
            info[machine_id] = {
                "total_gpus": len(machine.gpus),
                "available_gpus": len(machine.get_available_gpus()),
                "allocated_gpus": len([gpu for gpu in machine.gpus.values() if not gpu.available])
            }
        return info