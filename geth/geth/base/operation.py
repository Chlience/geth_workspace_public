from enum import Enum


class Operation:
    def __init__(self, op_code, args):
        self.op_code = op_code
        self.args = args

    def __str__(self):
        return f"{self.__class__}({self.op_code}, {self.args})"

    def __repr__(self):
        return str(self)


class SystemOperation(Operation):
    class OpCode(Enum):
        Timer = 0  # args: (current_timestamp,)

        ListAgent = 1  # args: ()
        ShowAgent = 2  # args: (agent_id)

        ListTask = 3  # args: ()
        CreateTask = 4  # args: (task_file, task_resources)
        ShowTask = 5  # args: (task_id)
        UpdateTask = 6  # args: (task_id, task_resources)
        DeleteTask = 7  # args: (task_id)

    def __init__(self, op_code: "SystemOperation.OpCode", args):
        super().__init__(op_code, args)


class HubOperation(Operation):
    class OpCode(Enum):
        Acknowledge = 0  # args: (result, response)
        InitWorker = 1  # args: (task_file, dist_info, wait_recover)
        TerminateWorker = 2  # args: ()
        PauseTask = 3  # args: ()
        UnpauseTask = 4  # args: (recovery_schedule, new_dist_info)
        PreScaleTask = 5  # args: (prescale_info)
        GetResources = 6  # args: (prescale_info)

    def __init__(self, op_code: "HubOperation.OpCode", args):
        super().__init__(op_code, args)


class AgentOperation(Operation):
    class OpCode(Enum):
        RegisterAgent = 0  # args: (agent_device, agent_ip, port)
        ReportAgentStatusPeriodic = 1  # args: (training_status)
        ReportAgentWorkerInitEvent = 2  # args: ()
        ReportAgentWorkerTerminateEvent = 3  # args: ()
        ReportAgentTrainingPauseEvent = 4  # args: ()
        ReportAgentTrainingUnpauseEvent = 5  # args: ()
        ReportAgentPreScaleEvent = 6  # args: ()
        ReportAgentGetResourcesEvent = 7  # args: ()

    def __init__(self, op_code: "AgentOperation.OpCode", args):
        super().__init__(op_code, args)


class DaemonOperation(Operation):
    class OpCode(Enum):
        InitWorker = 0  # args: ()
        StartTraining = 1  # args: (rec_flag, recover_schedule, dist_info)
        StopTraining = 2  # args: (save_flag)
        FiniWorker = 3  # args: ()
        QueryTrainingStatus = 4  # args: ()
        PreScaleTask = 5  # args: (prescale_info)
        QueryPreScaleStatus = 6  # args: ()

    def __init__(self, op_code: "DaemonOperation.OpCode", args):
        super().__init__(op_code, args)


class TrainerOperation(Operation):
    class OpCode(Enum):
        TrainingInited = 0
        TrainingTerminated = 1  # args: ()
        TrainingPaused = 2  # args: ()
        TrainingUnpaused = 3  # args: ()
        UpdateTrainerStatus = 4  # args: (TrainingStatus)
        UpdatePreScaleStatus = (
            5  # args: (one of "finished", "not_started", "in_progress")
        )

    def __init__(self, op_code: "TrainerOperation.OpCode", args):
        super().__init__(op_code, args)
