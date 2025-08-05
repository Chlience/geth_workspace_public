from enum import Enum


class TrainerStopReason(Enum):
    UNKNOWN = -1
    WAITING = 0
    FINISHED = 1
    STOPPED_BY_AGENT = 2


class TrainingStatus:
    def __init__(
        self,
        epoch: int,
        step: int,
        running: bool,
        stop_reason: TrainerStopReason,
        prescale_status: bool = False,
    ):
        self.epoch = epoch
        self.step = step
        self.running = running
        self.stop_reason = stop_reason
        self.prescale_status = prescale_status

    def __repr__(self) -> str:
        return f"TrainingStatus(epoch={self.epoch}, step={self.step}, running={self.running}, stop_reason={self.stop_reason}, prescale_status={self.prescale_status})"
