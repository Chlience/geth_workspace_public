import time

from loguru import logger

from geth.agent.worker import GethWorker
from geth.base.operation import (
    AgentOperation,
    DaemonOperation,
    HubOperation,
    TrainerOperation,
)
from geth.base.zmq_link import ZmqClient
from geth.utils.interval_checker import IntervalChecker

import subprocess

SLEEP_INTERVAL = 0.05


class GethAgent:
    def __init__(
        self, agnet_str: str, device_id: int, ip_address: str, port: int, endpoint: str
    ):
        self.agent_id = agnet_str
        self.worker_id = int(agnet_str)
        self.device_id = device_id
        self.ip_address = ip_address
        self.port = port
        self.zmq_client = ZmqClient(self.agent_id)
        self.stop = False
        self.zmq_client.connect(endpoint)
        self.worker = None

        self.training_status_report_checker = IntervalChecker(3)

        logger.info(
            f"Agent {self.agent_id} created with device {self.device_id} on {self.ip_address}:{self.port}"
        )

    def register(self):
        self.zmq_client.send(
            AgentOperation(
                AgentOperation.OpCode.RegisterAgent,
                (
                    self.device_id,
                    self.ip_address,
                    self.port,
                ),
            )
        )
        msg = self.zmq_client.recv()
        logger.debug(f"Registration response: {msg}")
        assert isinstance(msg, HubOperation)
        assert msg.op_code == HubOperation.OpCode.Acknowledge
        assert len(msg.args) == 2
        assert msg.args[0]
        logger.info("Registered with hub successfully.")

    def execute(self):
        self.register()
        while not self.stop:
            try:
                self.report_pre_scale_status()
                self.report_training_status()

                if self.zmq_client.has_data():
                    msg = self.zmq_client.recv()
                    self.handle_message(msg)
                else:
                    time.sleep(SLEEP_INTERVAL)
            except Exception as e:
                raise e

    def report_training_status(self):
        # just do report, all paused(stopped) to finished are send by hub
        if (
            self.worker is not None
            and self.worker.in_training_proc
            and not self.worker.in_pre_scale_proc
        ):
            if self.training_status_report_checker.is_time():
                self.worker.tx.put(
                    DaemonOperation(DaemonOperation.OpCode.QueryTrainingStatus, ())
                )
                resp = self.worker.rx.get()
                logger.debug(f"Training status: {resp.args[0]}")
                assert resp.op_code == TrainerOperation.OpCode.UpdateTrainerStatus
                if not resp.args[0].running:
                    self.worker.in_training_proc = False
                self.zmq_client.send(
                    AgentOperation(
                        AgentOperation.OpCode.ReportAgentStatusPeriodic,
                        (resp.args[0],),
                    )
                )
                self.training_status_report_checker.set_update()
                # handle stop in hub now:
                # 1. hub check for stop signal in ReportAgentStatusPeriodic from agent for every chain
                # 2. worker should issue stop when got request from hub

    def report_pre_scale_status(self):
        if (
            self.worker is not None
            # and self.worker.in_training_proc
            and self.worker.in_pre_scale_proc
        ):
            self.worker.tx.put(
                DaemonOperation(DaemonOperation.OpCode.QueryPreScaleStatus, ())
            )
            resp = self.worker.rx.get()
            logger.debug(f"PreScale status: {resp.args[0]}")
            assert resp.op_code == TrainerOperation.OpCode.UpdatePreScaleStatus
            if resp.args[0] == "in_process":
                return
            assert resp.args[0] == "finished"
            self.worker.in_pre_scale_proc = False
            self.zmq_client.send(
                AgentOperation(
                    AgentOperation.OpCode.ReportAgentPreScaleEvent,
                    (),
                )
            )

    def handle_message(self, op):
        assert isinstance(op, HubOperation)
        logger.info(f"Agent[{self.agent_id}] received hub operation: {op}")
        match op.op_code:
            case HubOperation.OpCode.InitWorker:
                task_name = op.args[0]
                task_file = op.args[1]
                dist_info = op.args[2]
                stay_paused = op.args[3] # Create: false, Scale: true
                target_epoch = op.args[4] if len(op.args) > 4 else -1
                self.worker = GethWorker(
                    self.worker_id, self.device_id, task_name, task_file
                )
                # recv worker init and report
                self.worker.tx.put(
                    DaemonOperation(
                        DaemonOperation.OpCode.InitWorker,
                        (dist_info.rank, dist_info.world_size, dist_info.master_worker_id, target_epoch),
                    )
                )
                resp = self.worker.rx.get()
                logger.debug(f"Trainer response for init task: {resp}")
                assert isinstance(resp, TrainerOperation)
                assert resp.op_code == TrainerOperation.OpCode.TrainingInited
                self.zmq_client.send(
                    AgentOperation(
                        AgentOperation.OpCode.ReportAgentWorkerInitEvent,
                        (), 
                    )
                )
                if not stay_paused:
                    waiting_for_device(self.device_id)
                    self.worker.tx.put(
                        DaemonOperation(
                            DaemonOperation.OpCode.StartTraining,
                            (
                                False,
                                None,
                                dist_info,
                            ),
                        )
                    )
                    self.worker.in_training_proc = True
                    resp = self.worker.rx.get()
                    logger.debug(f"Trainer response for start task: {resp}")
                    assert isinstance(resp, TrainerOperation)
                    assert resp.op_code == TrainerOperation.OpCode.TrainingUnpaused
                    self.zmq_client.send(
                        AgentOperation(
                            AgentOperation.OpCode.ReportAgentTrainingUnpauseEvent, ()
                        )
                    )

            case HubOperation.OpCode.PauseTask:
                assert self.worker is not None
                self.worker.tx.put(
                    DaemonOperation(DaemonOperation.OpCode.StopTraining, (True,))
                )
                resp = self.worker.rx.get()
                logger.debug(f"Agent[{self.agent_id}] got pause resp {resp}")
                assert isinstance(resp, TrainerOperation)
                assert resp.op_code == TrainerOperation.OpCode.TrainingPaused
                self.worker.in_training_proc = False
                self.zmq_client.send(
                    AgentOperation(
                        AgentOperation.OpCode.ReportAgentTrainingPauseEvent, ()
                    )
                )

            case HubOperation.OpCode.UnpauseTask:
                assert self.worker is not None
                rec_flag = op.args[0]
                recovery_schedule = op.args[1]
                dist_info = op.args[2]
                need_wait = op.args[3]
                if need_wait:
                    waiting_for_device(self.device_id)
                self.worker.tx.put(
                    DaemonOperation(
                        DaemonOperation.OpCode.StartTraining,
                        (
                            rec_flag,
                            recovery_schedule,
                            dist_info,
                        ),
                    )
                )
                self.worker.in_training_proc = True

                resp = self.worker.rx.get()
                logger.debug(f"Trainer response for recovery: {resp}")

                # later: check if these asserts hold true
                assert isinstance(resp, TrainerOperation)
                assert resp.op_code == TrainerOperation.OpCode.TrainingUnpaused
                self.zmq_client.send(
                    AgentOperation(
                        AgentOperation.OpCode.ReportAgentTrainingUnpauseEvent,
                        (True,),
                    )
                )

            case HubOperation.OpCode.TerminateWorker:
                assert self.worker is not None
                if self.worker.in_training_proc:
                    self.worker.tx.put(
                        DaemonOperation(DaemonOperation.OpCode.StopTraining, (True,))
                    )
                    resp = self.worker.rx.get()
                    logger.debug(f"Agent[{self.agent_id}] got terminate resp {resp}")
                    assert isinstance(resp, TrainerOperation)
                    assert resp.op_code == TrainerOperation.OpCode.TrainingPaused
                    self.worker.in_training_proc = False
                self.worker.tx.put(
                    DaemonOperation(DaemonOperation.OpCode.FiniWorker, ())
                )
                resp = self.worker.rx.get()
                logger.debug(f"Agent[{self.agent_id}] got fini resp {resp}")
                assert isinstance(resp, TrainerOperation)
                assert resp.op_code == TrainerOperation.OpCode.TrainingTerminated
                self.worker.stop()
                self.worker = None
                self.zmq_client.send(
                    AgentOperation(
                        AgentOperation.OpCode.ReportAgentWorkerTerminateEvent,
                        (),
                    )
                )
                logger.info(f"Agent[{self.agent_id}] exit.")
                exit(0)
            
            case HubOperation.OpCode.PreScaleTask:
                assert self.worker is not None
                prescale_info = op.args[0]
                self.worker.tx.put(
                    DaemonOperation(
                        DaemonOperation.OpCode.PreScaleTask,
                        (prescale_info,),
                    )
                )
                self.worker.in_pre_scale_proc = True
            
            case HubOperation.OpCode.GetResources:
                waiting_for_device(self.device_id)
                self.zmq_client.send(
                    AgentOperation(
                        AgentOperation.OpCode.ReportAgentGetResourcesEvent,
                        (), 
                    )
                )
            
            case _:
                raise NotImplementedError(f"Operation not implemented: {op.op_code}")


def waiting_for_device(device_id: int):
    while True:
        cmd = f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i {device_id}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()

        # Parse the output
        utilization, memory = map(int, output.split(","))

        logger.info(
            f"GPU {device_id}: Utilization={utilization}%, Memory Used={memory}MB"
        )

        # Check the condition
        if utilization == 0 and memory < 2048:
            logger.info(
                f"GPU {device_id} is idle. Proceeding..."
            )
            break
        time.sleep(1)