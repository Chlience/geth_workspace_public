import os
import glob
import json
import re
from datetime import datetime
import argparse

CREATE_TASK = re.compile(r"Requesting: <class 'geth\.base\.operation\.SystemOperation'>\(OpCode\.CreateTask")
INIT_TASK = re.compile(r"All agent inited for task Task")
REPORT_TRAIN_UNPAUSE_TASK = re.compile(r"<class 'geth\.base\.operation\.AgentOperation'>\(OpCode\.ReportAgentTrainingUnpauseEvent, \(\)\)")
AFTER_STEP_STAT = re.compile(r"geth\.trainer\.base_trainer:after_step:\d+ - Stat:")  # 133 → \d+
PREPARTITION_DONE = re.compile(r"geth\.trainer\.elastic_ddp_gnn_trainer:_partation_graph_and_distribute:\d+ - Prepartition done")
LOAD_PARTITION_DONE = re.compile(r"geth\.trainer\.elastic_ddp_gnn_trainer:_partation_graph_and_distribute:\d+ - Load prepartition done")
UPDATE_TASK = re.compile(r"<class 'geth\.base\.operation\.SystemOperation'>\(OpCode\.UpdateTask")
UNPAUSE_TASK = re.compile(r"<class 'geth\.base\.operation\.HubOperation'>\(OpCode\.UnpauseTask, \(True, GethDDPRecoveryShedule")
RECOVERY_GROUP_SETUP = re.compile(r"Training recovery group setup done")
DDP_PART_RECOVERY_START = re.compile(r"Start recovery of DDP Part")
TRAINER_RESPONSE_RECOVERY = re.compile(
    r"geth\.agent\.agent:handle_message:\d+ - Trainer response for recovery: <class 'geth\.base\.operation\.TrainerOperation'>\(OpCode\.TrainingUnpaused, \(\)\)"
)

def handle_no_scale_file(filepath):
    print(f"Handling file: {filepath}")
    with open(filepath, "r") as f:
        lines = f.readlines()

    results = {}
    results['raw'] = {}
    pre_step_durations = []
    last_time = None
    last_epoch_step = None
    part_load_finish_time = None

    found_create = found_init = found_report = False

    for line in lines:
        ts = line.split(" | ")[0].strip()
        have_time_in_log = False
        try:
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
            have_time_in_log = True
        except:
            have_time_in_log = False

        if have_time_in_log and not found_create and CREATE_TASK.search(line):
            create_time = t
            found_create = True
            continue
        if have_time_in_log and found_create and not found_init and INIT_TASK.search(line):
            inited_time = t
            found_init = True
            continue
        if have_time_in_log and found_init and not found_report and REPORT_TRAIN_UNPAUSE_TASK.search(line):
            report_time = t
            found_report = True
            continue
        if have_time_in_log and AFTER_STEP_STAT.search(line):
            m = re.search(r"epoch=(\d+), self\.step=(\d+)", line)
            if m:
                ep, st = int(m.group(1)), int(m.group(2))
                if last_time and (ep, st) != last_epoch_step:
                    pre_step_durations.append((t - last_time).total_seconds())
                last_time = t
                last_epoch_step = (ep, st)
            continue
        if PREPARTITION_DONE.search(line):
            m = re.search(r"waiting time: ([\d\.]+)s", line)
            if m:
                results["raw"]["prepartition_wait_time"] = float(m.group(1))
            continue
        if LOAD_PARTITION_DONE.search(line):
            m = re.search(r"time: ([\d\.]+)s", line)
            if m:
                results["raw"]["load_prepartition_time"] = float(m.group(1))
            part_load_finish_time = t
            continue

    results["raw"]["pre_training_step_durations"] = pre_step_durations
    results["worker_init_time"] = (inited_time - create_time).total_seconds()
    results["training_setup_time"] = (report_time - inited_time).total_seconds()
    if part_load_finish_time is not None:
        results["graph_related_time"] = {}
        results["graph_related_time"]["prepartition_wait_time"] = results["raw"]["prepartition_wait_time"]
        results["graph_related_time"]["load_prepartition_time"] = results["raw"]["load_prepartition_time"]
        results["graph_related_time"]["transfer_time"] = (report_time - part_load_finish_time).total_seconds()
    results["avg_time"] = sum(pre_step_durations) / len(pre_step_durations) if pre_step_durations else 0

    del results["raw"]

    key = os.path.splitext(os.path.basename(filepath))[0]
    
    return key, results


def handle_scale_file(filepath):
    print(f"Handling file: {filepath}")
    with open(filepath, "r") as f:
        lines = f.readlines()

    results = {}
    results['raw'] = {}
    pre_step_durations = []
    last_time = None
    last_epoch_step = None
    recovery_step_durations = []
    last_time_rec = None
    last_epoch_step_rec = None

    found_create = found_init = found_report = found_update = found_unpause = False

    for line in lines:
        ts = line.split(" | ")[0].strip()
        have_time_in_log = False
        try:
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
            have_time_in_log = True
        except:
            have_time_in_log = False

        if have_time_in_log and not found_create and CREATE_TASK.search(line):
            results["raw"]["create_task_time"] = t
            found_create = True
            continue
        if have_time_in_log and found_create and not found_init and INIT_TASK.search(line):
            results["raw"]["all_agent_inited_time"] = t
            found_init = True
            continue
        if have_time_in_log and found_init and not found_report and REPORT_TRAIN_UNPAUSE_TASK.search(line):
            results["raw"]["report_unpause_time"] = t
            found_report = True
            continue
        if have_time_in_log and not found_unpause:
            if UPDATE_TASK.search(line):
                results["raw"]["update_task_time"] = t
                found_update = True
                continue
            if AFTER_STEP_STAT.search(line):
                m = re.search(r"epoch=(\d+), self.step=(\d+)", line)
                if m:
                    ep, st = int(m.group(1)), int(m.group(2))
                    if last_time and (ep, st) != last_epoch_step:
                        pre_step_durations.append((t - last_time).total_seconds())
                    last_time = t
                    last_epoch_step = (ep, st)
                continue
        if have_time_in_log and found_update and not found_unpause and UNPAUSE_TASK.search(line):
            results["raw"]["hub_unpause_time"] = t
            found_unpause = True
            continue
        if found_unpause:
            if PREPARTITION_DONE.search(line):
                m = re.search(r"waiting time: ([\d\.]+)s", line)
                if m:
                    results["raw"]["prepartition_wait_time"] = float(m.group(1))
                continue
            if LOAD_PARTITION_DONE.search(line):
                m = re.search(r"time: ([\d\.]+)s", line)
                if m:
                    results["raw"]["load_prepartition_time"] = float(m.group(1))
                continue
            if have_time_in_log and RECOVERY_GROUP_SETUP.search(line):
                results["raw"]["recovery_group_setup_time"] = t
                continue
            if have_time_in_log and DDP_PART_RECOVERY_START.search(line):
                results["raw"]["start_recovery_time"] = t
                continue
            if have_time_in_log and TRAINER_RESPONSE_RECOVERY.search(line):
                results["raw"]["training_unpause_time"] = t
                continue
            if have_time_in_log and AFTER_STEP_STAT.search(line):
                m = re.search(r"epoch=(\d+), self.step=(\d+)", line)
                if m:
                    ep, st = int(m.group(1)), int(m.group(2))
                    if last_time_rec and (ep, st) != last_epoch_step_rec:
                        recovery_step_durations.append((t - last_time_rec).total_seconds())
                    last_time_rec = t
                    last_epoch_step_rec = (ep, st)
                continue

    results["raw"]["pre_training_step_durations"] = pre_step_durations
    results["raw"]["recovery_step_durations"] = recovery_step_durations

    results["worker_init_time"] = (results["raw"]["all_agent_inited_time"] - results["raw"]["create_task_time"]).total_seconds()
    results["training_setup_time"] = (results["raw"]["report_unpause_time"] - results["raw"]["all_agent_inited_time"]).total_seconds()
    results["avg_time_before_scale"] = sum(pre_step_durations) / len(pre_step_durations)

    results["overlapped_scale_time"] = (results["raw"]["hub_unpause_time"] - results["raw"]["update_task_time"]).total_seconds()
    results["non_overlapped_scale_time"] = {} 
    results["non_overlapped_scale_time"]["total"] = (results["raw"]["training_unpause_time"] - results["raw"]["hub_unpause_time"]).total_seconds()
    results["non_overlapped_scale_time"]["setup_graph_and_ddp_group"] = (results["raw"]["recovery_group_setup_time"] - results["raw"]["hub_unpause_time"]).total_seconds()
    if "start_recovery_time" in results["raw"]:
        results["non_overlapped_scale_time"]["prepartition_wait_time"] = results["raw"]["prepartition_wait_time"]
        results["non_overlapped_scale_time"]["load_prepartition_time"] = results["raw"]["load_prepartition_time"]
        results["non_overlapped_scale_time"]["graph_data_transfer"] = (results["raw"]["start_recovery_time"] - results["raw"]["recovery_group_setup_time"]).total_seconds()
        results["non_overlapped_scale_time"]["ddp_data_transfer"] = (results["raw"]["training_unpause_time"] - results["raw"]["start_recovery_time"]).total_seconds()
    else:
        results["non_overlapped_scale_time"]["ddp_data_transfer"] = (results["raw"]["training_unpause_time"] - results["raw"]["recovery_group_setup_time"]).total_seconds()

    results["avg_time_after_scale"] = sum(recovery_step_durations) / len(recovery_step_durations)

    del results["raw"]

    key = os.path.splitext(os.path.basename(filepath))[0]

    return key, results


def handle_folder(folder_path):
    print(f"Handling folder: {folder_path}")
    pattern = os.path.join(folder_path, '*.txt')
    txt_files = glob.glob(pattern)
    
    results = {
        "scale": {},
        "no_scale": {}
    }
    
    for file in txt_files:
        basename = os.path.basename(file)
        if basename.startswith("scale"):
            k, v = handle_scale_file(file)
            results["scale"][k] = v
        elif basename.startswith("no_scale"):
            k, v = handle_no_scale_file(file)
            results["no_scale"][k] = v
    
    results["scale"] = dict(sorted(results["scale"].items()))
    results["no_scale"] = dict(sorted(results["no_scale"].items()))

    with open(os.path.join(folder_path, "results.json"), "w") as f:
        # custom default: datetime -> timestamp, else str
        json.dump(results, f, indent=2,
                  default=lambda o: o.timestamp() if isinstance(o, datetime) else Exception())
    
    return results

def save_result_for_plot(results, ckpt_time):
    # calc overlapped time and non overlapped
    # calc how much iter its take as overhead before
    new_res = {}

    for k, v in results['scale'].items():
        from_proc = int(k.split("_")[1])
        to_proc = int(k.split("_")[2])

        base_no_scale = f"no_scale_no_micro_{to_proc}"
        if base_no_scale not in results['no_scale']:
            base_no_scale = f"no_scale_{to_proc}"

        naive_overhead = results['no_scale'][base_no_scale]["worker_init_time"] + results['no_scale'][base_no_scale]["training_setup_time"] + ckpt_time
        naive_iters = naive_overhead / v["avg_time_before_scale"]

        scale_overhead_overlapped = results['scale'][k]["overlapped_scale_time"]
        if "prepartition_wait_time" in results['scale'][k]["non_overlapped_scale_time"]:
            scale_overhead_overlapped += results['scale'][k]["non_overlapped_scale_time"]["prepartition_wait_time"]
        
        scale_overhead_non_overlapped = results['scale'][k]["non_overlapped_scale_time"]["total"]
        if "prepartition_wait_time" in results['scale'][k]["non_overlapped_scale_time"]:
            scale_overhead_non_overlapped -= results['scale'][k]["non_overlapped_scale_time"]["prepartition_wait_time"]
        
        scale_overlapped_iters = scale_overhead_overlapped / v["avg_time_before_scale"]
        scale_non_overlapped_iters = scale_overhead_non_overlapped / v["avg_time_before_scale"]

        at_least_benefitial_iters = scale_overhead_non_overlapped / (v["avg_time_before_scale"] - v["avg_time_after_scale"])
    
        new_res[k] = {
            "naive_iters": naive_iters,
            "naive_overhead": naive_overhead,
            "scale_overlapped_iters": scale_overlapped_iters,
            "scale_overhead_overlapped": scale_overhead_overlapped,
            "scale_non_overlapped_iters": scale_non_overlapped_iters,
            "scale_overhead_non_overlapped": scale_overhead_non_overlapped,
            "at_least_benefitial_iters": at_least_benefitial_iters
        }

    return new_res

def main():
    parser = argparse.ArgumentParser(description="Process GETH results.")
    parser.add_argument("--ckpt_perf", type=str, default="ckpt_perf.json",
                        help="Path to the checkpoint performance JSON file.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing the results folders.")
    parser.add_argument("--output_file", type=str, default="overall_results.json",
                        help="Output file for overall results.")
    args = parser.parse_args()
    
    with open(args.ckpt_perf, "r") as f:
        ckpt_res = json.load(f)
        ckpt_res_map = {
            "gat": ckpt_res["GAT"],
            "gcn": ckpt_res["GCN"],
            "gin": ckpt_res["GIN"],
            "sage": ckpt_res["SAGE"],
        }
    
    # Scan the results folder and get all directories
    results_dir = args.results_dir
    result_folders = []
    
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path):
                result_folders.append(item)
    
    overall_res = {}
    for folder in result_folders:
        if folder.startswith("."):
            continue
        res = handle_folder(os.path.join(results_dir, folder))
        task_type = folder.split("_")[0]
        ckpt_time = ckpt_res_map[task_type]['save'] + ckpt_res_map[task_type]['load']
        overall_res[folder] = save_result_for_plot(res, ckpt_time)

    with open(args.output_file, "w") as f:
        json.dump(overall_res, f, indent=2,
                  default=lambda o: o.timestamp() if isinstance(o, datetime) else Exception())
    
if __name__ == "__main__":
    main()
