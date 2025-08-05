"""
Core simulation engine implementation.
"""
import time
from typing import List, Optional, Dict, Any
import math
import simpy
import attr
from loguru import logger
from ..cluster.resource import ClusterState, GPU, get_gpu_ids
from ..jobs.job import Job, JobStatus
from ..scheduler.base import Scheduler, SJFScheduler, FIFOScheduler, ElasticScheduler, ElasticPrestartScheduler, SchedulerMetrics, ElasticPrestartSjfScheduler

enable_cluster_info = False
enable_task_info = False

import simpy.rt

@attr.s(auto_attribs=True)
class SimulationConfig:
    """Configuration for simulation parameters."""
    seed: int = 42
    simulation_duration: float = 3600  # seconds
    scheduler_interval: float = 0.5  # seconds
    enable_visualization: bool = False
    vi_url: str = "http://localhost:5000"

@attr.s(auto_attribs=True)
class SimulationMetrics:
    """Metrics collected during simulation."""
    total_runtime: float = 0.0
    scheduler_matrics: SchedulerMetrics = None

class SimulationEngine:
    """Main simulation engine."""
    
    def __init__(self, config: SimulationConfig, scheduler_type: str = "fifo"):
        self.config = config
        if self.config.enable_visualization:
            self.env = simpy.rt.RealtimeEnvironment(initial_time=0, factor=1, strict=False)
        else:
            # self.env = simpy.rt.RealtimeEnvironment(initial_time=0, factor=0.2, strict=False)
            self.env = simpy.Environment()
        self.termination = self.env.event()
        self.cluster = ClusterState()
        
        # Initialize scheduler based on type
        if scheduler_type.lower() == "elastic_pre":
            self.scheduler: Scheduler = ElasticPrestartScheduler(self.env)
        elif scheduler_type.lower() == "elastic_pre_sjf":
            self.scheduler: Scheduler = ElasticPrestartSjfScheduler(self.env)
        elif scheduler_type.lower() == "elastic":
            self.scheduler: Scheduler = ElasticScheduler(self.env)
        elif scheduler_type.lower() == "sjf":
            self.scheduler: Scheduler = SJFScheduler(self.env)
        else:  # Default to FIFO
            self.scheduler: Scheduler = FIFOScheduler(self.env)
            
        self.scheduler.set_cluster(self.cluster)
        self.metrics = SimulationMetrics()
        self.all_job_submitted = self.env.event()

        # if self.config.enable_visualization:
        #     self.vi_connector = VIConnector(self.env, self.scheduler, self.config.vi_url, self.all_job_submitted)
        
        # Track pending job submissions
        self.pending_job_submissions = {}
        
    def add_machine(self, machine_id: str, num_gpus: int) -> None:
        self.cluster.add_machine(machine_id, num_gpus)
            
    def submit_jobs(self, jobs: List[Job]) -> None:
        for job in jobs:
            self.scheduler.submit_job(job)
            
    def submit_job_at_time(self, job: Job, arrival_time: float) -> None:
        def _delayed_submission():
            yield self.env.timeout(arrival_time)
            self.scheduler.submit_job(job)
            logger.debug(f"{job}")
            logger.debug(f"[{self.env.now:.2f}]: Job {job.job_id} submitted")
            # Remove from pending submissions once submitted
            if job.job_id in self.pending_job_submissions:
                del self.pending_job_submissions[job.job_id]
            if len(self.pending_job_submissions) == 0:
                self.all_job_submitted.succeed()
            
        # Record this job as pending submission
        self.pending_job_submissions[job.job_id] = arrival_time
        self.env.process(_delayed_submission())
            
    def _scheduler_process(self) -> None:
        while True:
            self.scheduler.schedule()
            if enable_cluster_info:
                logger.debug(f"[{self.env.now:.2f}]: cluster_info={self.cluster.get_cluster_info()}")
            if enable_task_info:
                logger.debug(f"[{self.env.now:.2f}]: task_info={self.scheduler.get_queue_task_info()}")
            yield self.env.timeout(self.config.scheduler_interval)
            
    def run(self) -> SimulationMetrics:
        """Run the simulation."""
        # Start scheduler process
        self.env.process(self._scheduler_process())
        
        # Start job execution processes for running jobs
        def job_monitor():
            last_check_time = self.env.now
            no_activity_duration = 0
            
            while True:
                # Check for running jobs that need processes
                activity_detected = False
                
                for job in list(self.scheduler.running_jobs.values()):
                    if job.status == JobStatus.RUNNING:
                        activity_detected = True
                
                # Check if there are any jobs in the queue, running, or pending submission
                if self.scheduler.running_jobs or self.scheduler.queue or self.pending_job_submissions:
                    activity_detected = True
                
                # If no activity and reached simulation duration, terminate
                if not activity_detected:
                    no_activity_duration += (self.env.now - last_check_time)
                    if no_activity_duration >= 10 or self.env.now >= self.config.simulation_duration:
                        logger.info(f"[{self.env.now:.2f}]: No activity detected for {no_activity_duration} seconds, terminating simulation")
                        yield self.termination.succeed()
                        break
                else:
                    no_activity_duration = 0
                
                last_check_time = self.env.now
                
                # Check if simulation duration has been reached
                if self.env.now >= self.config.simulation_duration:
                    logger.info(f"[{self.env.now:.2f}]: Simulation duration reached")
                    yield self.termination.succeed()
                    break
                    
                yield self.env.timeout(1)
                
        self.env.process(job_monitor())

        # if self.config.enable_visualization:
        #     self.env.process(self.vi_connector.run())
        
        # Run simulation
        self.env.run(until=self.termination)

        # if self.config.enable_visualization:
        #     self.vi_connector.send_cluster_snapshot(finished=True)
        #     time.sleep(1)
        
        # Update final metrics
        self.metrics.scheduler_matrics = self.scheduler.get_metrics()
        self.metrics.total_runtime = self.env.now
        
        return self.metrics
