import time


class IntervalChecker:
    def __init__(self, update_interval: int):
        """
        Initialize the updater with a specific update interval.

        :param update_interval: Interval in seconds to determine when an update is needed
        """
        self.update_interval = update_interval
        self.last_update_time = time.time()

    def is_time(self):
        """
        Check if an update is needed based on the update interval.

        :return: Boolean indicating if an update is needed
        """
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            return True
        return False

    def set_update(self):
        """
        Reset the last update time to the current time.
        """
        self.last_update_time = time.time()

    def __repr__(self):
        return (
            f"IntervalChecker(update_interval={self.update_interval}, "
            f"last_update_time={self.last_update_time})"
        )
