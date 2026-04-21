import os
import json
import time
import math
import argparse
import heapq
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set
from scipy.optimize import linear_sum_assignment



@dataclass
class InstanceLB:
    name: str
    num_jobs: int
    num_machines: int
    num_agv: int
    lu_node: int
    rw_node: int
    pt: List[List[int]]      
    pt_rw: List[List[int]]   
    tau: np.ndarray
    setup: List[List[int]]   
    routings: Dict[int, List[int]]
    disassembly_job_ids: Set[int]

@dataclass(order=True)
class Task:
    q_neg: float 
    id: str = field(compare=False)
    r: float = field(compare=False)
    p: float = field(compare=False)
    q: float = field(compare=False)


def solve_jackson(task_dicts: List[Dict], num_machines: int = 1) -> float:
    if not task_dicts: return 0.0
    tasks = [Task(q_neg=-float(t['q']), id=t['id'], r=float(t['r']), p=float(t['p']), q=float(t['q'])) for t in task_dicts]
    unreleased = sorted(tasks, key=lambda x: x.r)
    current_time = unreleased[0].r if unreleased else 0.0
    ready_queue = [] 
    max_cmax = 0.0
    
    while ready_queue or unreleased:
        while unreleased and unreleased[0].r <= current_time + 1e-6:
            heapq.heappush(ready_queue, unreleased.pop(0))
        if not ready_queue and unreleased:
            current_time = unreleased[0].r
            continue
        active = []
        while len(active) < num_machines and ready_queue:
            active.append(heapq.heappop(ready_queue))
        
        next_rel = unreleased[0].r if unreleased else float('inf')
        min_fin = min((current_time + t.p for t in active), default=float('inf'))
        next_evt = min(next_rel, min_fin)
        dt = next_evt - current_time
        current_time = next_evt
        
        for t in active:
            t.p -= dt
            if t.p <= 1e-6: max_cmax = max(max_cmax, current_time + t.q)
            else: heapq.heappush(ready_queue, t)
    return max_cmax

def calc_min_empty_travel(agv_tasks, num_agv, tau, lu_node):
    if not agv_tasks: return 0
    supply = [t['dest'] for t in agv_tasks] + [lu_node] * num_agv
    demand = [t['orig'] for t in agv_tasks] + [-1] * num_agv
    
    dim = len(supply)
    cost_mat = np.zeros((dim, dim))
    for r in range(dim):
        for c in range(dim):
            if demand[c] == -1:
                cost_mat[r][c] = 0.0 
            else:
                cost_mat[r][c] = tau[supply[r]][demand[c]]
                
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    return sum(cost_mat[r][c] for r, c in zip(row_ind, col_ind))

def run_lower_bound(inst: InstanceLB):
    heads = {j: [0]*len(inst.routings[j]) for j in range(inst.num_jobs)}
    tails = {j: [0]*len(inst.routings[j]) for j in range(inst.num_jobs)}
    
    for j in range(inst.num_jobs):
        route = inst.routings[j]
        t = inst.tau[inst.lu_node][route[0]] 
        for i, node in enumerate(route):
            if i > 0: 
                t += inst.tau[route[i-1]][node]
            heads[j][i] = t
            p = 0
            if 1 <= node <= inst.num_machines:
                p = inst.pt[node-1][j] 
            t += p

    for j in range(inst.num_jobs):
        route = inst.routings[j]
        is_dis = j in inst.disassembly_job_ids
        
        t_res = inst.tau[route[-1]][inst.lu_node] 
        
        for i in range(len(route)-1, -1, -1):
            node = route[i]
            p_node = 0
            if 1 <= node <= inst.num_machines:
                p_node = inst.pt[node-1][j] 

            if not is_dis:
                tails[j][i] = t_res
                if i > 0: 
                    t_res = inst.tau[route[i-1]][node] + p_node + tails[j][i]
            else:
                tail_core = 0
                if 1 <= node <= inst.num_machines:
                    p_rw = inst.pt_rw[node-1][j]
                    tail_core = inst.tau[node][inst.rw_node] + p_rw + inst.tau[inst.rw_node][inst.lu_node]
                tails[j][i] = max(t_res, tail_core)
                if i > 0: 
                    t_res = inst.tau[route[i-1]][node] + p_node + tails[j][i]

    agv_tasks, rw_tasks = [], []
    sw_tasks = {m: [] for m in range(1, inst.num_machines + 1)}
    LB_Job = 0
    
    for j in range(inst.num_jobs):
        route = inst.routings[j]
        is_dis = j in inst.disassembly_job_ids
        
        p_in = inst.tau[inst.lu_node][route[0]]
        q_in = (inst.pt[route[0]-1][j] if 1 <= route[0] <= inst.num_machines else 0) + tails[j][0]
        agv_tasks.append({'id': f"In_J{j}", 'r': 0, 'p': p_in, 'q': q_in, 'orig': inst.lu_node, 'dest': route[0]})
        
        for i, curr in enumerate(route):
            p_raw = 0
            if 1 <= curr <= inst.num_machines:
                p_raw = inst.pt[curr-1][j]
            
            LB_Job = max(LB_Job, heads[j][i] + p_raw + tails[j][i])
            
            if 1 <= curr <= inst.num_machines:
                sw_tasks[curr].append({'id': f"J{j}", 'r': heads[j][i], 'p': p_raw, 'q': tails[j][i]})
                
            if i < len(route) - 1:
                next_node = route[i+1]
                r_t = heads[j][i] + p_raw 
                p_t = inst.tau[curr][next_node]
                q_t = tails[j][i+1] + (inst.pt[next_node-1][j] if 1 <= next_node <= inst.num_machines else 0)
                agv_tasks.append({'id': f"Move_J{j}", 'r': r_t, 'p': p_t, 'q': q_t, 'orig': curr, 'dest': next_node})
            else:
                r_out = heads[j][i] + p_raw
                p_out = inst.tau[curr][inst.lu_node]
                agv_tasks.append({'id': f"Out_J{j}", 'r': r_out, 'p': p_out, 'q': 0, 'orig': curr, 'dest': inst.lu_node})
                
            if is_dis and 1 <= curr <= inst.num_machines:
                p_rw = inst.pt_rw[curr-1][j]
                r1 = heads[j][i] + p_raw
                p1 = inst.tau[curr][inst.rw_node]
                q1 = p_rw + inst.tau[inst.rw_node][inst.lu_node]
                
                rw_tasks.append({'id': f"RW_J{j}", 'r': r1+p1, 'p': p_rw, 'q': inst.tau[inst.rw_node][inst.lu_node]})
                agv_tasks.append({'id': f"Core_J{j}", 'r': r1, 'p': p1, 'q': q1, 'orig': curr, 'dest': inst.rw_node})
                agv_tasks.append({'id': f"Ret_J{j}", 'r': r1+p1+p_rw, 'p': inst.tau[inst.rw_node][inst.lu_node], 'q': 0, 'orig': inst.rw_node, 'dest': inst.lu_node})

    LB_SW = max([solve_jackson(sw_tasks[m]) for m in sw_tasks], default=0)
    LB_RW = solve_jackson(rw_tasks)
    LB_W = max(LB_SW, LB_RW)
    
    total_loaded = sum(t['p'] for t in agv_tasks)
    min_empty = calc_min_empty_travel(agv_tasks, inst.num_agv, inst.tau, inst.lu_node)
    
    LB_AGV_Cap = math.ceil((total_loaded + min_empty) / inst.num_agv) if inst.num_agv > 0 else 0
    LB_AGV_Sched = solve_jackson(agv_tasks, num_machines=inst.num_agv)
    LB_AGV = max(LB_AGV_Cap, LB_AGV_Sched)
    
    return {
        "LB_Job": float(LB_Job), 
        "LB_W": float(LB_W), 
        "LB_AGV": float(LB_AGV), 
        "Final_LB": float(max(LB_Job, LB_W, LB_AGV))
    }


class LBLoader:
    @staticmethod
    def load(filename: str, data: dict) -> InstanceLB:
        M, J = data['M'], data['J']
        
        pt_matrix = [[0] * J for _ in range(M)]
        for m_idx in range(M):
            for j_idx in range(J):
                if j_idx < len(data['pt'][m_idx]):
                    pt_matrix[m_idx][j_idx] = data['pt'][m_idx][j_idx]

        pt_rw_matrix = [[0] * J for _ in range(M)]
        nja = data['nja']
        disassembly_ids = set()
        
        for j_id in range(nja + 1, J + 1):
            disassembly_ids.add(j_id - 1)
            rel_idx = j_id - (nja + 1)
            for m_idx in range(M):
                if rel_idx < len(data['rpt'][m_idx]):
                    pt_rw_matrix[m_idx][j_id-1] = data['rpt'][m_idx][rel_idx]

        routings = {}
        for j_id in range(1, J + 1):
            route_raw = data['routes'][str(j_id)]
            routings[j_id-1] = [int(x) for x in route_raw]

        return InstanceLB(
            name=filename, num_jobs=J, num_machines=M, num_agv=data['fleet'],
            lu_node=0, rw_node=M + 1, pt=pt_matrix, pt_rw=pt_rw_matrix,
            tau=np.array(data['tau']), setup=data.get('setup', []),
            routings=routings, disassembly_job_ids=disassembly_ids
        )

def evaluate_single_instance(file_path: str, output_dir: str):
    target_file = Path(file_path)
    out_path = Path(output_dir)
    
    if not target_file.exists():
        print(f"[Error] Instance file '{target_file}' not found.")
        return

    print(f"Initiating analytical evaluation for instance: {target_file.name}")
    
    try:
        with open(target_file, 'r') as f:
            data = json.load(f)
        
        inst = LBLoader.load(target_file.name, data)
        
        t0 = time.time()
        res = run_lower_bound(inst)
        cpu_time = time.time() - t0
        
        output = {
            "instance": inst.name.replace(".json", ""),
            "size_jobs": inst.num_jobs,
            "LB_Final": res['Final_LB'],
            "LB_components": res,
            "LB_CPU": round(cpu_time, 4)
        }
            
        target_out_path = out_path / f"{target_file.stem}_LB.json"
        target_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_out_path, 'w') as f_out:
            json.dump(output, f_out, indent=4)
            
        print(f"Evaluation Successful. Final LB: {res['Final_LB']:.2f}")
        print(f"Results archived at: {target_out_path}")

    except Exception as e:
        print(f"[Error] Analytical failure on {target_file.name}: {str(e)}")

def run_batch_campaign(input_dir: str, output_dir: str):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    if not in_path.exists():
        print(f"[Error] Directory '{in_path}' not found")
        return

    files = list(in_path.rglob("*.json"))
    if not files:
        print("[Error] No JSON instances in the specified directory")
        return

    print(f"Starting LB computation for {len(files)} instances...")
    t_start = time.time()
    
    for idx, f_path in enumerate(files, 1):
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
            
            inst = LBLoader.load(f_path.name, data)
            
            t0 = time.time()
            res = run_lower_bound(inst)
            cpu_time = time.time() - t0
            
            output = {
                "instance": inst.name.replace(".json", ""),
                "size_jobs": inst.num_jobs,
                "LB_Final": res['Final_LB'],
                "LB_components": res,
                "LB_CPU": round(cpu_time, 4)
            }
            
            rel_path = f_path.relative_to(in_path)
            target_path = out_path / rel_path.with_name(f"{f_path.stem}_LB.json")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_path, 'w') as f_out:
                json.dump(output, f_out, indent=4)
                
            if idx % 100 == 0:
                elapsed = time.time() - t_start
                print(f"[{time.strftime('%H:%M:%S')}] Processed {idx}/{len(files)} (Elapsed: {elapsed:.1f}s)")

        except Exception as e:
            print(f"[Error] Failed on {f_path.name}: {str(e)}")

    print(f"\nFinished batch processing in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive theoretical lower bounds for the 3SMRSP-T.")
    parser.add_argument(
        "--instance", 
        type=str, 
        help="Path to a specific JSON instance for isolated evaluation. If omitted, full batch execution is triggered."
    )
    
    args = parser.parse_args()

    DEFAULT_INPUT_DIR = "instances"
    DEFAULT_OUTPUT_DIR = "results/lower_bounds"

    if args.instance:
        evaluate_single_instance(args.instance, DEFAULT_OUTPUT_DIR)
    else:
        run_batch_campaign(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)