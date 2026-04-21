import json
import random
import math
import time
import os
import argparse
import sys
import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path


ALGORITHM_PARAMETERS = {
    'ALNS': {
        'reaction_factor': 0.5,
        'segment_length': 50,
        'destruction_rate': 0.4,
        'initial_temperature_ratio': 0.01,
    },
    'MNSA': {
        'cooling_rate': 0.98,
        'cutoff_ratio': 0.6,
        'initial_temperature_ratio': 0.05,
    }
}

EXECUTION_CONFIG = {
    'random_seed': 1994,
    'default_input_dir': "instances",
    'default_output_dir': "data/results/metaheuristics"
}


@dataclass
class JobParameters:
    identifier: int
    is_manufacturing: bool
    technological_route: List[int]
    processing_durations: Dict[int, int]
    reprocessing_durations: Dict[int, int]

@dataclass
class ProblemInstance:
    n_workstations: int
    n_jobs: int
    n_vehicles: int
    setup_matrix: List[List[int]]
    distance_matrix: List[List[int]]
    jobs: Dict[int, JobParameters]
    reprocessing_node: int
    load_unload_node: int = 0

class InstanceReader:
    @staticmethod
    def read_json(data: dict) -> ProblemInstance:
        n_machines, n_jobs = data['M'], data['J']
        rw_node = n_machines + 1
        jobs = {}

        for j_id in range(1, data['nja'] + 1):
            route = [int(x) for x in data['routes'][str(j_id)]]
            pt = {m: data['pt'][m-1][j_id-1] for m in range(1, n_machines+1)}
            jobs[j_id] = JobParameters(j_id, True, route, pt, {})

        for j_id in range(data['nja'] + 1, n_jobs + 1):
            route = [int(x) for x in data['routes'][str(j_id)]]
            pt = {m: data['pt'][m-1][j_id-1] for m in range(1, n_machines+1)}
            rel_idx = j_id - (data['nja'] + 1)
            rpt = {m: data['rpt'][m-1][rel_idx] for m in range(1, n_machines+1) if data['rpt'][m-1][rel_idx] > 0}
            jobs[j_id] = JobParameters(j_id, False, route, pt, rpt)

        setup = data.get('setup', [[0] * n_jobs for _ in range(n_jobs)])
        tau   = data.get('tau',   [[0] * (n_machines+2) for _ in range(n_machines+2)])
        
        return ProblemInstance(
            n_workstations=n_machines, 
            n_jobs=n_jobs, 
            n_vehicles=data['fleet'], 
            setup_matrix=setup, 
            distance_matrix=tau, 
            jobs=jobs, 
            reprocessing_node=rw_node
        )


@njit(cache=False)
def _compute_system_exit_time(
    sequence: np.ndarray, total_operations: int, n_machines: int, n_jobs: int, n_agvs: int,
    lu_node: int, rw_node: int, distance_matrix: np.ndarray, setup_matrix: np.ndarray, 
    routes: np.ndarray, route_lengths: np.ndarray, processing_times: np.ndarray, 
    reprocessing_times: np.ndarray, is_forward_flow: np.ndarray
) -> float:
    
    mach_free = np.zeros(n_machines + 2, dtype=np.float64)
    mach_last = np.full(n_machines + 2, -1, dtype=np.int32)
    agv_free  = np.zeros(n_agvs, dtype=np.float64)
    agv_loc   = np.full(n_agvs, lu_node, dtype=np.int32)
    job_idx   = np.zeros(n_jobs, dtype=np.int32)
    job_avail = np.zeros(n_jobs, dtype=np.float64)

    cores_rw_j    = np.zeros(total_operations, dtype=np.int32)
    cores_rw_dest = np.zeros(total_operations, dtype=np.int32)
    cores_rw_cr   = np.zeros(total_operations, dtype=np.float64)
    n_cores_rw    = 0

    cores_lu_j  = np.zeros(total_operations, dtype=np.int32)
    cores_lu_cr = np.zeros(total_operations, dtype=np.float64)
    n_cores_lu  = 0

    pending_idx = 0
    n_pending   = len(sequence)
    ops_completed = 0
    current_time  = 0.0

    INF = 1e15

    while ops_completed < total_operations or n_cores_rw > 0 or n_cores_lu > 0:
        best_arrival = INF
        c_agv_pickup = 0.0; c_end_time = 0.0; c_event_type = -1; c_job_id = -1; c_agv_id = -1
        c_orig = -1; c_dest = -1; c_list_idx = -1; c_start_proc = 0.0

        if pending_idx < n_pending:
            j     = sequence[pending_idx]
            ri    = job_idx[j]
            orig  = lu_node if ri == 0 else routes[j, ri - 1]
            dest  = lu_node if ri == route_lengths[j] else routes[j, ri]
            ready = job_avail[j]
            d     = distance_matrix[orig, dest]

            ba = INF; bp = 0.0; bv = -1
            for v in range(n_agvs):
                p = max(agv_free[v] + distance_matrix[agv_loc[v], orig], ready)
                a = p + d
                if a < ba:
                    ba = a; bp = p; bv = v

            proc = ba
            dur  = 0.0
            if dest != lu_node:
                lj = mach_last[dest]
                su = setup_matrix[lj, j] if (lj != -1 and dest != rw_node) else 0.0
                proc = max(ba, mach_free[dest] + su)
                dur = processing_times[j, dest]

            if ba < best_arrival or (ba == best_arrival and bp < c_agv_pickup):
                best_arrival = ba
                c_agv_pickup = bp; c_end_time = proc + dur; c_event_type = 0; c_job_id = j
                c_agv_id = bv; c_orig = orig; c_dest = dest; c_list_idx = 0; c_start_proc = proc

        for i in range(n_cores_rw):
            cj = cores_rw_j[i]
            co = cores_rw_dest[i]
            cr = cores_rw_cr[i]
            d  = distance_matrix[co, rw_node]

            ba = INF; bp = 0.0; bv = -1
            for v in range(n_agvs):
                p = max(agv_free[v] + distance_matrix[agv_loc[v], co], cr)
                a = p + d
                if a < ba:
                    ba = a; bp = p; bv = v

            s = max(ba, mach_free[rw_node])
            if ba < best_arrival or (ba == best_arrival and bp < c_agv_pickup):
                best_arrival = ba
                c_agv_pickup = bp; c_end_time = s + reprocessing_times[cj, co]; c_event_type = 1; c_job_id = cj
                c_agv_id = bv; c_orig = co; c_dest = rw_node; c_list_idx = i; c_start_proc = s

        for i in range(n_cores_lu):
            cj = cores_lu_j[i]
            cr = cores_lu_cr[i]
            d  = distance_matrix[rw_node, lu_node]

            ba = INF; bp = 0.0; bv = -1
            for v in range(n_agvs):
                p = max(agv_free[v] + distance_matrix[agv_loc[v], rw_node], cr)
                a = p + d
                if a < ba:
                    ba = a; bp = p; bv = v

            if ba < best_arrival or (ba == best_arrival and bp < c_agv_pickup):
                best_arrival = ba
                c_agv_pickup = bp; c_end_time = ba; c_event_type = 2; c_job_id = cj
                c_agv_id = bv; c_orig = rw_node; c_dest = lu_node; c_list_idx = i; c_start_proc = ba

        if c_event_type == -1:
            nxt = INF
            for m in range(n_machines + 2):
                if current_time < mach_free[m] < nxt: nxt = mach_free[m]
            for v in range(n_agvs):
                if current_time < agv_free[v] < nxt: nxt = agv_free[v]
            if pending_idx < n_pending:
                p_j = sequence[pending_idx]
                if current_time < job_avail[p_j] < nxt: nxt = job_avail[p_j]
            for i in range(n_cores_rw):
                if current_time < cores_rw_cr[i] < nxt: nxt = cores_rw_cr[i]
            for i in range(n_cores_lu):
                if current_time < cores_lu_cr[i] < nxt: nxt = cores_lu_cr[i]

            if nxt >= INF: break
            current_time = nxt
            continue

        agv_loc[c_agv_id]  = c_dest
        agv_free[c_agv_id] = best_arrival
        if c_agv_pickup > current_time: current_time = c_agv_pickup

        if c_event_type == 0:
            if c_dest != lu_node:
                mach_free[c_dest] = c_end_time
                mach_last[c_dest] = c_job_id
                job_avail[c_job_id] = c_end_time
                if is_forward_flow[c_job_id] == 0 and reprocessing_times[c_job_id, c_dest] > 0:
                    cores_rw_j[n_cores_rw]    = c_job_id
                    cores_rw_dest[n_cores_rw] = c_dest
                    cores_rw_cr[n_cores_rw]   = c_end_time
                    n_cores_rw += 1
            else:
                job_avail[c_job_id] = c_end_time
            job_idx[c_job_id] += 1
            pending_idx  += 1
            ops_completed += 1
            
        elif c_event_type == 1:
            mach_free[rw_node] = c_end_time
            last = n_cores_rw - 1
            if c_list_idx != last:
                cores_rw_j[c_list_idx]    = cores_rw_j[last]
                cores_rw_dest[c_list_idx] = cores_rw_dest[last]
                cores_rw_cr[c_list_idx]   = cores_rw_cr[last]
            n_cores_rw -= 1
            
            cores_lu_j[n_cores_lu]  = c_job_id
            cores_lu_cr[n_cores_lu] = c_end_time
            n_cores_lu += 1
            
        elif c_event_type == 2:
            last = n_cores_lu - 1
            if c_list_idx != last:
                cores_lu_j[c_list_idx]  = cores_lu_j[last]
                cores_lu_cr[c_list_idx] = cores_lu_cr[last]
            n_cores_lu -= 1

    final_makespan = 0.0
    for m in range(n_machines + 2):
        if mach_free[m] > final_makespan: final_makespan = mach_free[m]
    for v in range(n_agvs):
        if agv_free[v] > final_makespan: final_makespan = agv_free[v]
        
    return final_makespan

class ScheduleDecoder:
    def __init__(self, instance: ProblemInstance):
        self.total_operations = sum(len(j.technological_route) + 1 for j in instance.jobs.values())
        self.n_workstations = instance.n_workstations
        self.n_jobs = instance.n_jobs
        self.n_vehicles = instance.n_vehicles
        self.node_lu = instance.load_unload_node
        self.node_rw = instance.reprocessing_node

        self.job_list = sorted(instance.jobs.keys())
        self.job_to_index = {j_id: i for i, j_id in enumerate(self.job_list)}

        self.setup_matrix = np.array(instance.setup_matrix, dtype=np.float64)
        self.distance_matrix = np.array(instance.distance_matrix, dtype=np.float64)

        max_route_length = max(len(j.technological_route) for j in instance.jobs.values())
        self.routes = np.zeros((self.n_jobs, max_route_length), dtype=np.int32)
        self.route_lengths = np.zeros(self.n_jobs, dtype=np.int32)

        num_nodes = self.n_workstations + 2
        self.processing_times = np.zeros((self.n_jobs, num_nodes), dtype=np.float64)
        self.reprocessing_times = np.zeros((self.n_jobs, num_nodes), dtype=np.float64)
        self.is_forward_flow = np.zeros(self.n_jobs, dtype=np.int32)

        for j_id, job in instance.jobs.items():
            idx = self.job_to_index[j_id]
            self.route_lengths[idx] = len(job.technological_route)
            for i, node in enumerate(job.technological_route):
                self.routes[idx, i] = node
            for m, t in job.processing_durations.items():
                self.processing_times[idx, m] = t
            for m, t in job.reprocessing_durations.items():
                self.reprocessing_times[idx, m] = t
            self.is_forward_flow[idx] = 1 if job.is_manufacturing else 0

    def evaluate_sequence(self, sequence: List[int]) -> float:
        seq_array = np.array([self.job_to_index[j] for j in sequence], dtype=np.int32)
        return _compute_system_exit_time(
            seq_array, self.total_operations, self.n_workstations, self.n_jobs, self.n_vehicles,
            self.node_lu, self.node_rw, self.distance_matrix, self.setup_matrix,
            self.routes, self.route_lengths, self.processing_times, self.reprocessing_times, self.is_forward_flow
        )


class AdaptiveLargeNeighborhoodSearch:
    def __init__(self, problem_instance: ProblemInstance, initial_sequence: List[int], parameters: Dict):
        self.instance = problem_instance
        self.evaluator = ScheduleDecoder(problem_instance)
        
        self.global_best_sequence = list(initial_sequence)
        self.global_best_objective = self.evaluator.evaluate_sequence(self.global_best_sequence)
        self.current_sequence = list(self.global_best_sequence)
        self.current_objective = self.global_best_objective
        
        self.reaction_factor = parameters['reaction_factor']
        self.segment_length = parameters['segment_length']
        self.destruction_rate = parameters['destruction_rate']
        self.start_temperature_ratio = parameters['initial_temperature_ratio']
        
        self.score_rewards = {'global_best': 33, 'better': 9, 'accepted': 13, 'rejected': 0}
        
        self.removal_operators = [self.remove_random, self.remove_worst_cost, self.remove_related, self.remove_sequence]
        self.insertion_operators = [self.insert_greedy, self.insert_regret_2]
        
        n_rem, n_ins = len(self.removal_operators), len(self.insertion_operators)
        self.removal_weights, self.insertion_weights = [1.0] * n_rem, [1.0] * n_ins
        self.removal_scores, self.insertion_scores = [0.0] * n_rem, [0.0] * n_ins
        self.removal_usage, self.insertion_usage = [0] * n_rem, [0] * n_ins
        
        self.rng = random.Random()
        
        self.job_resources = {j: set(job.processing_durations.keys()) for j, job in problem_instance.jobs.items()}
        self.job_workloads = {
            j: sum(job.processing_durations.values()) + sum(job.reprocessing_durations.values()) 
            for j, job in problem_instance.jobs.items()
        }
        
        self.workload_range = max(1, max(self.job_workloads.values()) - min(self.job_workloads.values()))
        self.relatedness_matrix = self._compute_shaw_relatedness()

    def _compute_shaw_relatedness(self):
        matrix = {}
        job_identifiers = list(self.instance.jobs.keys())
        for j1 in job_identifiers:
            matrix[j1] = {}
            for j2 in job_identifiers:
                if j1 == j2: 
                    matrix[j1][j2] = 1.0
                    continue
                intersection = len(self.job_resources[j1] & self.job_resources[j2])
                union = len(self.job_resources[j1] | self.job_resources[j2])
                machine_similarity = intersection / union if union > 0 else 0
                workload_difference = abs(self.job_workloads[j1] - self.job_workloads[j2]) / self.workload_range
                matrix[j1][j2] = (0.7 * machine_similarity) - (0.3 * workload_difference)
        return matrix

    def remove_related(self, sequence, q_elements):
        if q_elements >= len(sequence): return [], list(sequence)
        seed_job = sequence[self.rng.randint(0, len(sequence)-1)]
        scores = [(self.relatedness_matrix.get(seed_job, {}).get(j, 0.0) + self.rng.uniform(-0.1, 0.1), i) 
                  for i, j in enumerate(sequence) if sequence[i] != seed_job]
        scores.sort(key=lambda x: x[0], reverse=True)
        removal_indices = sorted([sequence.index(seed_job)] + [x[1] for x in scores[:q_elements-1]], reverse=True)
        
        partial_seq = list(sequence); extracted_jobs = []
        for i in removal_indices: extracted_jobs.append(partial_seq.pop(i))
        return partial_seq, extracted_jobs

    def remove_random(self, sequence, q_elements):
        partial_seq = list(sequence); extracted_jobs = []
        for _ in range(q_elements): extracted_jobs.append(partial_seq.pop(self.rng.randint(0, len(partial_seq)-1)))
        return partial_seq, extracted_jobs

    def remove_sequence(self, sequence, q_elements):
        if q_elements >= len(sequence): return [], list(sequence)
        start_idx = self.rng.randint(0, len(sequence)-q_elements)
        return sequence[:start_idx] + sequence[start_idx+q_elements:], sequence[start_idx:start_idx+q_elements]

    def remove_worst_cost(self, sequence, q_elements):
        if q_elements >= len(sequence): return [], list(sequence)
        target_set = set(sorted(sequence, key=lambda j: self.job_workloads[j] * self.rng.uniform(0.5, 1.5), reverse=True)[:q_elements])
        return [x for x in sequence if x not in target_set], [x for x in sequence if x in target_set]

    def _determine_sparse_neighborhood(self, current_length, start_time, max_time):
        elapsed = time.time() - start_time
        if max_time <= 0 or 1.0 - (elapsed / max_time) < 0.05: return [current_length], "APPEND"
        target_evaluations = 15 if 1.0 - (elapsed / max_time) < 0.2 else 40
        step_size = max(1, current_length // target_evaluations)
        indices = list(range(0, current_length + 1, step_size))
        if current_length not in indices: indices.append(current_length)
        return indices, "NORMAL"

    def insert_greedy(self, partial_sequence, pending_jobs, start_time, max_time):
        candidate_seq = list(partial_sequence)
        for job in sorted(pending_jobs, key=lambda j: self.job_workloads[j], reverse=True):
            indices, mode = self._determine_sparse_neighborhood(len(candidate_seq), start_time, max_time)
            if mode == "APPEND": 
                candidate_seq.append(job)
                continue
            best_position, best_cost = -1, float('inf')
            for i in indices:
                candidate_seq.insert(i, job)
                cost = self.evaluator.evaluate_sequence(candidate_seq)
                del candidate_seq[i]
                if cost < best_cost: best_cost, best_position = cost, i
            candidate_seq.insert(best_position if best_position != -1 else len(candidate_seq), job)
        return candidate_seq

    def insert_regret_2(self, partial_sequence, pending_jobs, start_time, max_time):
        candidate_seq = list(partial_sequence)
        unassigned = list(pending_jobs)
        while unassigned:
            indices, mode = self._determine_sparse_neighborhood(len(candidate_seq), start_time, max_time)
            if mode == "APPEND": return self.insert_greedy(candidate_seq, unassigned, start_time, max_time)
            
            insertion_candidates = []
            for job in sorted(unassigned, key=lambda j: self.job_workloads[j], reverse=True)[:5]:
                costs = []
                for i in indices:
                    candidate_seq.insert(i, job)
                    costs.append((self.evaluator.evaluate_sequence(candidate_seq), i))
                    del candidate_seq[i]
                costs.sort(key=lambda x: x[0])
                insertion_candidates.append((job, costs[0][1], costs[0][0], (costs[1][0] if len(costs) > 1 else costs[0][0] * 1.5) - costs[0][0]))
                
            best_decision = max(insertion_candidates, key=lambda x: (x[3], -x[2]))
            unassigned.remove(best_decision[0])
            candidate_seq.insert(best_decision[1], best_decision[0])
        return candidate_seq

    def select_roulette_wheel(self, weights):
        total_weight = sum(weights)
        if total_weight <= 0: return self.rng.randint(0, len(weights)-1)
        spin, cumulative = self.rng.uniform(0, total_weight), 0.0
        for i, weight in enumerate(weights):
            cumulative += weight
            if cumulative > spin: return i
        return len(weights)-1

    def update_operator_weights(self):
        rf = self.reaction_factor
        for i in range(len(self.removal_operators)):
            if self.removal_usage[i] > 0: self.removal_weights[i] = max(0.05, (1 - rf) * self.removal_weights[i] + rf * (self.removal_scores[i] / self.removal_usage[i]))
            self.removal_scores[i], self.removal_usage[i] = 0, 0
        for i in range(len(self.insertion_operators)):
            if self.insertion_usage[i] > 0: self.insertion_weights[i] = max(0.05, (1 - rf) * self.insertion_weights[i] + rf * (self.insertion_scores[i] / self.insertion_usage[i]))
            self.insertion_scores[i], self.insertion_usage[i] = 0, 0

    def optimize(self, computational_budget: float, random_seed: int):
        self.rng.seed(random_seed)
        start_time = time.time()
        iterations, stagnation_counter = 0, 0
        temperature, final_temperature = max(self.global_best_objective * self.start_temperature_ratio, 1.0), 0.01
        
        while time.time() - start_time < computational_budget:
            rem_idx, ins_idx = self.select_roulette_wheel(self.removal_weights), self.select_roulette_wheel(self.insertion_weights)
            dynamic_ratio = self.destruction_rate if stagnation_counter < 50 else min(0.7, self.destruction_rate * 1.5)
            
            partial_seq, extracted = self.removal_operators[rem_idx](self.current_sequence, self.rng.randint(4, max(5, int(len(self.current_sequence) * dynamic_ratio))))
            neighbor_candidate = self.insertion_operators[ins_idx](partial_seq, extracted, start_time, computational_budget)
            
            if not neighbor_candidate or len(neighbor_candidate) != len(self.current_sequence): break 
            
            candidate_objective = self.evaluator.evaluate_sequence(neighbor_candidate)
            status = 'rejected'
            
            if candidate_objective < self.global_best_objective:
                self.global_best_objective, self.global_best_sequence = candidate_objective, list(neighbor_candidate)
                status = 'global_best'
                self.current_sequence, self.current_objective = neighbor_candidate, candidate_objective
                stagnation_counter = 0
            elif candidate_objective < self.current_objective:
                status = 'better'
                self.current_sequence, self.current_objective = neighbor_candidate, candidate_objective
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if self.rng.random() < math.exp(-(candidate_objective - self.current_objective) / max(temperature, 1e-5)):
                    status = 'accepted'
                    self.current_sequence, self.current_objective = neighbor_candidate, candidate_objective
                    
            reward = self.score_rewards[status]
            self.removal_scores[rem_idx] += reward
            self.insertion_scores[ins_idx] += reward
            self.removal_usage[rem_idx] += 1
            self.insertion_usage[ins_idx] += 1
            
            iterations += 1
            if iterations % self.segment_length == 0: self.update_operator_weights()
            temperature = max(final_temperature, (self.global_best_objective * self.start_temperature_ratio) * math.pow(final_temperature / max(1.0, self.global_best_objective * self.start_temperature_ratio), min(1.0, (time.time() - start_time) / computational_budget)))
            
        return self.global_best_objective, self.global_best_sequence


class MultiNeighborhoodSimulatedAnnealing:
    def __init__(self, problem_instance: ProblemInstance, initial_sequence: List[int], parameters: Dict):
        self.instance = problem_instance
        self.evaluator = ScheduleDecoder(problem_instance)
        
        self.global_best_sequence = list(initial_sequence)
        self.global_best_objective = self.evaluator.evaluate_sequence(self.global_best_sequence)
        self.current_sequence = list(self.global_best_sequence)
        self.current_objective = self.global_best_objective
        
        self.rng = random.Random()
        self.cooling_rate = parameters['cooling_rate']
        self.cutoff_ratio = parameters['cutoff_ratio']
        self.start_temperature_ratio = parameters['initial_temperature_ratio']
        self.final_temperature = 0.01
        self.neighborhood_operators = (self.apply_swap_operator, self.apply_shift_operator, self.apply_reverse_operator)

    def apply_swap_operator(self, sequence: List[int]) -> List[int]:
        n_elements = len(sequence)
        if n_elements < 2: return list(sequence)
        perturbed_sequence = list(sequence)
        idx_1, idx_2 = self.rng.randrange(n_elements), self.rng.randrange(n_elements)
        perturbed_sequence[idx_1], perturbed_sequence[idx_2] = perturbed_sequence[idx_2], perturbed_sequence[idx_1]
        return perturbed_sequence

    def apply_shift_operator(self, sequence: List[int]) -> List[int]:
        n_elements = len(sequence)
        if n_elements < 2: return list(sequence)
        perturbed_sequence = list(sequence)
        idx_from, idx_to = self.rng.randrange(n_elements), self.rng.randrange(n_elements)
        if idx_from == idx_to: return perturbed_sequence
        shifted_job = perturbed_sequence.pop(idx_from); perturbed_sequence.insert(idx_to, shifted_job)
        return perturbed_sequence

    def apply_reverse_operator(self, sequence: List[int]) -> List[int]:
        n_elements = len(sequence)
        if n_elements < 2: return list(sequence)
        idx_1, idx_2 = self.rng.randrange(n_elements), self.rng.randrange(n_elements)
        if idx_1 > idx_2: idx_1, idx_2 = idx_2, idx_1
        return sequence[:idx_1] + sequence[idx_1:idx_2+1][::-1] + sequence[idx_2+1:]

    def optimize(self, computational_budget: float, random_seed: int, target_iterations: int) -> Tuple[float, List[int]]:
        self.rng.seed(random_seed)
        start_time = time.time()

        initial_temperature = self.global_best_objective * self.start_temperature_ratio
        current_temperature = initial_temperature
        if self.final_temperature >= initial_temperature: self.final_temperature = initial_temperature * 0.001

        logarithmic_ratio = math.log(self.final_temperature / initial_temperature) / math.log(self.cooling_rate)
        plateau_length = max(1, int(target_iterations / logarithmic_ratio))
        acceptance_limit = int(self.cutoff_ratio * plateau_length)

        step_counter, accepted_counter = 0, 0
        random_float, select_operator, exp_function = self.rng.random, self.rng.choice, math.exp
        evaluate_makespan, operators = self.evaluator.evaluate_sequence, self.neighborhood_operators

        while time.time() - start_time < computational_budget:
            neighbor_candidate = select_operator(operators)(self.current_sequence)
            candidate_objective = evaluate_makespan(neighbor_candidate)
            objective_delta = candidate_objective - self.current_objective

            if objective_delta < 0 or random_float() < exp_function(-objective_delta / max(current_temperature, 1e-10)):
                self.current_sequence, self.current_objective = neighbor_candidate, candidate_objective
                accepted_counter += 1
                if candidate_objective < self.global_best_objective:
                    self.global_best_objective, self.global_best_sequence = candidate_objective, list(neighbor_candidate)

            step_counter += 1
            if step_counter >= plateau_length or accepted_counter >= acceptance_limit:
                current_temperature = max(current_temperature * self.cooling_rate, self.final_temperature)
                step_counter, accepted_counter, acceptance_limit = 0, 0, int(self.cutoff_ratio * plateau_length)

        return self.global_best_objective, self.global_best_sequence


def generate_initial_solution(problem_instance: ProblemInstance, evaluator: ScheduleDecoder, n_trials: int = 5) -> Tuple[float, List[int]]:
    rng = random.Random(EXECUTION_CONFIG['random_seed'])
    operations_count = {j: len(job.technological_route) + 1 for j, job in problem_instance.jobs.items()}
    best_sequence, best_objective = [], float('inf')
    
    for _ in range(n_trials):
        sequence, remaining_operations = [], operations_count.copy()
        available_jobs = list(remaining_operations.keys())
        
        while available_jobs:
            selected_job = rng.choice(available_jobs)
            sequence.append(selected_job)
            remaining_operations[selected_job] -= 1
            if remaining_operations[selected_job] == 0: available_jobs.remove(selected_job)
                
        objective_value = evaluator.evaluate_sequence(sequence)
        if objective_value < best_objective: best_objective, best_sequence = objective_value, sequence
            
    return best_objective, best_sequence

def calibrate_computational_throughput(problem_instance: ProblemInstance, evaluator: ScheduleDecoder, duration: float = 0.2) -> int:
    rng, operations_count = random.Random(EXECUTION_CONFIG['random_seed']), {j: len(job.technological_route) + 1 for j, job in problem_instance.jobs.items()}
    dummy_sequence, remaining_operations = [], operations_count.copy()
    available_jobs = list(remaining_operations.keys())
    
    while available_jobs:
        selected_job = rng.choice(available_jobs)
        dummy_sequence.append(selected_job)
        remaining_operations[selected_job] -= 1
        if remaining_operations[selected_job] == 0: available_jobs.remove(selected_job)

    start_time, evaluations_completed = time.time(), 0
    while time.time() - start_time < duration:
        evaluator.evaluate_sequence(dummy_sequence)
        evaluations_completed += 1
        
    actual_duration = time.time() - start_time
    return int(evaluations_completed / actual_duration) if actual_duration > 0 else 1



def evaluate_computational_instance(file_path: Path, output_directory: Path, input_directory: Path):
    if not file_path.exists(): return
    with open(file_path, 'r') as file_descriptor:
        instance = InstanceReader.read_json(json.load(file_descriptor))
        
    decoder = ScheduleDecoder(instance)
    random.seed(EXECUTION_CONFIG['random_seed'])
    initial_objective, initial_sequence = generate_initial_solution(instance, decoder)
    
    total_operations = sum(len(job.technological_route) + 1 for job in instance.jobs.values())
    time_budget = max(5.0, total_operations * 0.125)
    target_mnsa_iterations = int(calibrate_computational_throughput(instance, decoder, duration=0.2) * time_budget * 0.85)

    results_payload = {
        "instance_identifier": file_path.stem,
        "job_magnitude": instance.n_jobs,
        "baseline_initial_value": initial_objective,
        "computational_constraints": {"allocated_time_seconds": round(time_budget, 2)}
    }

    t0_alns = time.time()
    obj_alns, _ = AdaptiveLargeNeighborhoodSearch(instance, list(initial_sequence), ALGORITHM_PARAMETERS['ALNS']).optimize(computational_budget=time_budget, random_seed=EXECUTION_CONFIG['random_seed'])
    results_payload["ALNS"] = {"objective": obj_alns, "computational_overhead": round(time.time() - t0_alns, 3)}

    t0_mnsa = time.time()
    obj_mnsa, _ = MultiNeighborhoodSimulatedAnnealing(instance, list(initial_sequence), ALGORITHM_PARAMETERS['MNSA']).optimize(computational_budget=time_budget, random_seed=EXECUTION_CONFIG['random_seed'], target_iterations=target_mnsa_iterations)
    results_payload["MNSA"] = {"objective": obj_mnsa, "computational_overhead": round(time.time() - t0_mnsa, 3)}

    relative_architecture = file_path.relative_to(input_directory) if input_directory in file_path.parents else Path(file_path.name)
    target_output_path = output_directory / relative_architecture.with_name(f"{file_path.stem}_results.json")
    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(target_output_path, 'w') as output_descriptor:
        json.dump(results_payload, output_descriptor, indent=4)

if __name__ == "__main__":
    from pathlib import Path
    
    # ==============================================================================
    # EXECUTION SETTINGS
    # ==============================================================================
    
    # Set to True to evaluate a single instance. Set to False to evaluate the entire testbed.
    RUN_SINGLE_INSTANCE = True 
    
    # Paste the exact relative path to the JSON instance file here:
    TARGET_JSON_PATH = "test_instances\T7_13_HD_A1_D5_L4_V3_SL_Det.json"
    
    # ==============================================================================

    input_dir = Path(EXECUTION_CONFIG['default_input_dir'])
    output_dir = Path(EXECUTION_CONFIG['default_output_dir'])

    if RUN_SINGLE_INSTANCE:
        target_path = Path(TARGET_JSON_PATH)
        
        if not target_path.exists():
            print(f"[Error] The file {target_path} does not exist. Verify the relative path.")
        else:
            evaluate_computational_instance(target_path, output_dir, input_dir)
            
    else:
        if input_dir.exists():
            for file_path in input_dir.rglob("*.json"):
                try: 
                    evaluate_computational_instance(file_path, output_dir, input_dir)
                except Exception: 
                    pass