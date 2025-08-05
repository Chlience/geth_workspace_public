import time

class PerfCounter:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.timer_map = {}
        self.timer_list = []
        self.cur_name = "root"
        self.depth = 0
    
    def record_start(self, name):
        self.timer_map[name] = {
            "parent": self.cur_name,
            "start": time.time(),
            "depth": self.depth
        }
        self.cur_name = name
        self.depth += 1
        print(self.cur_name + " begin")
    
    def record_end(self, name):
        self.timer_map[name]["end"] = time.time()
        self.cur_name = self.timer_map[name]["parent"]
        self.depth -= 1
        self.timer_list.append(self.timer_map[name])

    def summary(self):
        print("Start time: ", self.start_time)
        for name, timer in self.timer_map.items():
            print(f"{timer['depth'] * ' '} {timer['depth']} {name}: {timer['end'] - timer['start']}s @{timer['start']}")

perf_counter = PerfCounter()
