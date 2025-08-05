import math
import simpy
from loguru import logger
from .job import Job, JobStatus
from simpy import Event
from simpy.events import AllOf
from ..cluster.resource import get_gpu_ids

class JobProc:
    def __init__(self, env, job: Job, scheduler) -> None:
        self.env = env
        self.job = job
        self.scheduler = scheduler
        self.action = env.process(self.run())
        self.current_speed = 0
        self.pending_scale = None
    
    def run(self):
        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} starting job execution, in {get_gpu_ids(self.job.assigned_gpus)}")
        # todo: simulate setup time
        self.job.update_status(JobStatus.ALLOCATED, self.env.now)


        yield self.env.timeout(self.job.get_env_setup_time_with_partition())
        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} finished worker initialization")

        assigned_gpus_jobs = [g.current_job for g in self.job.assigned_gpus if (not g.available and g.current_job != self.job.job_id)]
        pending_events = [self.scheduler.running_jobs[j].job_finished_event for j in assigned_gpus_jobs if j in self.scheduler.running_jobs]
        yield AllOf(self.env, pending_events)
        for g in self.job.assigned_gpus:
            if g.current_job != self.job.job_id:
                g.commit_pre_alloc()
            assert g.current_job == self.job.job_id
        self.job.current_gpus = self.job.assigned_gpus.copy()
        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} finished waiting for assigned gpus")
        self.job.update_status(JobStatus.RUNNING, self.env.now)

        self.current_speed = self.job.get_job_speed()
        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} speed calculated: {self.current_speed:.2f}")
        self.job.record_performance(self.current_speed)
        self.job.record_gpu_allocation(self.env.now)

        # logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} starting execution in {get_gpu_ids(self.job.assigned_gpus)}")

        while self.job.status != JobStatus.COMPLETED:
            try:
                while self.job.status != JobStatus.COMPLETED:
                    if self.pending_scale is not None and self.pending_scale['apply_after'] <= self.env.now:
                        # PART 2: Pause training for a certain duration to simulate the job restart
                        if len(self.job.current_gpus)<=len(self.job.assigned_gpus):
                            assigned_gpus_jobs = [g.current_job for g in self.job.assigned_gpus if (not g.available and g.current_job != self.job.job_id)]
                            pending_events = [self.scheduler.running_jobs[j].job_finished_event for j in assigned_gpus_jobs if j in self.scheduler.running_jobs]
                            yield AllOf(self.env, pending_events)
                            for g in self.job.assigned_gpus:
                                if g.current_job != self.job.job_id:
                                    g.commit_pre_alloc()
                                assert g.current_job == self.job.job_id
                        else:
                            for g in self.job.current_gpus:
                                if g not in self.job.assigned_gpus:
                                    g.release()
                        pause_time = self.job.get_scale_pause_time()
                        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} pausing for {pause_time:.2f}s during scaling operation")
                        self.job.update_status(JobStatus.SCALING, self.env.now)
                        yield self.env.timeout(pause_time)

                        self.job.current_gpus = self.job.assigned_gpus.copy()
                        self.job.update_status(JobStatus.RUNNING, self.env.now)
                        
                        self.current_speed = self.job.get_job_speed()
                        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} recover after scaling operation, running at new speed: {self.current_speed:.2f} with {len(self.job.current_gpus)} GPUs")
                        self.job.record_performance(self.current_speed)
                        self.job.record_gpu_allocation(self.env.now)
                        
                        self.pending_scale = None
                        # logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} recover after scaling operation")

                    iteration_time = 1.0 / self.current_speed
                    yield self.env.timeout(iteration_time)
                    if self.job.samples_per_epoch == 1:
                        self.job.current_sample += 1
                    else:
                        self.job.current_sample += self.job.batch_size * len(self.job.current_gpus)

                    if self.job.current_sample >= self.job.samples_per_epoch:
                        self.job.current_sample = 0
                        self.job.current_epoch += 1

                    if self.job.current_epoch == self.job.target_epoch:
                        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} finished")
                        self.scheduler.complete_job(self.job, success=True)
                        self.job.update_status(JobStatus.COMPLETED, self.env.now)
                    
            except simpy.Interrupt as interrupt:
                self.handle_interrupt(interrupt)

    def handle_interrupt(self, interrupt):
        cause = interrupt.cause
        assert cause == "scale"
        if self.job.current_gpus == self.job.assigned_gpus:
            return

        env_setup_time = self.job.get_scale_overlapped_time()
        self.pending_scale = { # todo: change to event
            "apply_after": self.env.now + env_setup_time
        }
        self.job.update_status(JobStatus.PRE_SCALING, self.env.now)
        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} scale from {get_gpu_ids(self.job.current_gpus)} to {get_gpu_ids(self.job.assigned_gpus)}")
        logger.debug(f"[{self.env.now:.2f}]: Job {self.job.job_id} continuing at original speed {self.job.get_job_speed():.2f} with {len(self.job.current_gpus)} GPUs, for {env_setup_time:.2f}s during environment setup")


    def trigger_scale(self):
        self.action.interrupt("scale")