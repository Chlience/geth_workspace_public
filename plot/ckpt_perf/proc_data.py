#!/usr/bin/env python3
import os
import glob
import re
import json

# Mapping from filename prefix to subplot title
model_mapping = {
    'gcn': 'GCN',
    'gat': 'GAT',
    'gin': 'GIN',
    'sage': 'SAGE',
}

# Directory containing result txt files
base_dir = os.path.join(os.path.dirname(__file__), 'ckpt_perf_res')
files = sorted(glob.glob(os.path.join(base_dir, '*.txt')))

results = {}
for fpath in files:
    fname = os.path.basename(fpath)
    key = fname.rsplit('.', 1)[0]
    title = model_mapping.get(key, key)
    save_time = None
    load_time = None
    bcast = {}
    raw_bcast = {}
    with open(fpath) as f:
        for line in f:
            if line.startswith('[SaveCkpt]'):
                m = re.search(r'avg=([0-9.]+)s', line)
                if m:
                    save_time = float(m.group(1))
            elif line.startswith('[LoadCkpt]'):
                m = re.search(r'avg=([0-9.]+)s', line)
                if m:
                    load_time = float(m.group(1))
            elif '[BroadcastSerialized]' in line:
                m = re.search(r'world_size=(\d+), avg=([0-9.]+)s', line)
                if m:
                    ws = int(m.group(1))
                    tm = float(m.group(2))
                    bcast[ws] = tm
            elif '[RawBroadcast]' in line:
                m = re.search(r'world_size=(\d+), avg=([0-9.]+)s', line)
                if m:
                    ws = int(m.group(1))
                    tm = float(m.group(2))
                    raw_bcast[ws] = tm
    # sort world sizes
    ws_sorted = sorted(bcast.keys())
    raw_sorted = sorted(raw_bcast.keys())
    results[title] = {
        'save': save_time,
        'load': load_time,
        'bcast': {'sizes': ws_sorted, 'times': [bcast[w] for w in ws_sorted]},
        'raw': {'sizes': raw_sorted, 'times': [raw_bcast[w] for w in raw_sorted]}
    }

# save results
with open(f"ckpt_perf.json", "w") as f:
    json.dump(results, f, indent=2)