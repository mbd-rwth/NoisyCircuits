import math
import os
import time
import warnings
import numpy as np
from collections import deque
from joblib import Parallel, delayed


INTERACTIONS = [
    "nearest_neighbour",
    "all_to_all"
]

_CUTOFF_KEYS = (
    "single",
    "double",
    "readout"
)

_LOCAL_OPERATORS = {
    "id" : (np.eye(2, dtype=np.complex128), ),
    "i" : (np.eye(2, dtype=np.complex128),),
    "x" : (np.array([[0, 1], [1, 0]], dtype=np.complex128), ),
    "y" : (np.array([[0, -1j], [1j, 0]], dtype=np.complex128), ),
    "z" : (np.array([[1, 0], [0, -1]], dtype=np.complex128), ), 
    "reset" : (
                np.array([[1, 0], [0, 0]], dtype=np.complex128),
                np.array([[0, 1], [0, 0]], dtype=np.complex128)
            )
}

_LOG_FIDELITY_FLOOR = np.log(1e-12)


def log_fidelity(error : float) -> float:
    return math.log1p(-error) if error < 1 - 1e-12 else _LOG_FIDELITY_FLOOR

def _average_log_infidelity(
        process_fidelity : float,
        d : int
    ) -> float:
    return max(0.0, 1.0 - (d * process_fidelity  + 1.0) / (d + 1.0))

def channel_infidelity(kraus_ops : list[np.ndarray]) -> float:
    d = kraus_ops[0].shape[0]
    overlap = np.sum(np.abs(np.trace(k)) ** 2 for k in kraus_ops)
    return max(0.0, 1.0 - (overlap + d) / (d * (d + 1.0)))

def readout_infidelity(confusion_matrix : np.ndarray) -> float:
    confusion_matrix = np.asarray(confusion_matrix, dtype=np.float64)
    return 0.5 * (confusion_matrix[0, 1] + confusion_matrix[1, 0])

def select_and_relabel(
        single_qubit_errors : dict,
        two_qubit_errors : dict,
        measurement_errors : dict,
        connectivity : dict,
        coupling_map : list[tuple[int, int]],
        layout : list[int]
    ) -> tuple[dict, dict, dict, dict, list[tuple[int, int]]]:
    if len(set(layout)) != len(layout):
        raise ValueError("Layout contains duplicate physical qubits.")
    p2l = {
        p : i for i, p in enumerate(layout)
    }
    single = {
        i : single_qubit_errors[p] for i, p in enumerate(layout) if p in single_qubit_errors
    }
    double = {
        gate : {(p2l[a], p2l[b]) : v for (a,b), v in pairs.items() if a in p2l and b in p2l} for gate, pairs in two_qubit_errors.items()
    }
    measurement = {
        i : measurement_errors[p] for i,p in enumerate(layout) if p in measurement_errors
    }
    conn = {
        i : [p2l[x] for x in connectivity[p] if x in p2l] for i,p in enumerate(layout) if p in connectivity
    }
    coupling = [
        (p2l[a], p2l[b]) for a,b in coupling_map if a in p2l and b in p2l
    ]
    return single, double, measurement, conn, coupling

def connected_components(
        connectivity : dict
    ) -> list[list[int]]:
    seen, components = set(), []
    for start in connectivity:
        if start in seen:
            continue
        seen.add(start)
        component, stack = [start], [start]
        while stack:
            for v in connectivity[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    component.append(v)
                    stack.append(v)
        components.append(sorted(component))
    return components

def remove_couplers(
        connectivity : dict,
        coupling_map : list[tuple[int, int]],
        couplers : list[tuple[int, int]]
    ) -> tuple[dict, list[tuple[int, int]]]:
    removed = {frozenset(c) for c in couplers}
    conn = {
        q : [x for x in nnbs if frozenset((q, x)) not in removed] for q, nnbs in connectivity.items()
    }
    coupling = [
        c for c in coupling_map if frozenset(c) not in removed
    ]
    return conn, coupling

class _ProcessFidelityCalculator:
    def __init__(self) -> None:
        self.cache = {}

    @staticmethod
    def operators(
        name : str,
        params : list[np.ndarray]
    ) -> tuple:
        if name == "kraus":
            return tuple(np.array([[complex(*e) for e in row] for row in matrix], dtype=np.complex128) for matrix in params)
        if name not in _LOCAL_OPERATORS:
            raise ValueError("Unsuported Noise Instruction : {}".format(name))
        return _LOCAL_OPERATORS[name]

    def chain(
            self,
            steps : tuple
    ) -> float:
        key = tuple(step[:2] for step in steps)
        if key not in self.cache:
            kraus = [np.eye(2, dtype=np.complex128)]
            for name, _, params in steps:
                kraus = [op @ k for op in self.operators(name, params) for k in kraus]
            self.cache[key] = sum(abs(np.trace(k)) ** 2 for k in kraus) / 4.0
        return self.cache[key]

    @staticmethod
    def step(
        name : str,
        params : list[np.ndarray]
    ) -> tuple:
        if name == "kraus":
            return (name, id(params), params)
        return (name.lower(), None, None)

    def single_qubit_term(
            self,
            term : list[dict]
    ) -> float:
        return self.chain(tuple(self.step(g["name"], g.get("params")) for g in term))

    def two_qubit_term(
            self,
            term : list[dict]
    ) -> float:
        pauli, local = {}, {}
        for g in term:
            if g["name"] == "pauli":
                label = (g.get("params") or ["II"])[0]
                for i, q in enumerate(g["qubits"]):
                    if i < len(label):
                        pauli[q] = label[i]
            elif len(g["qubits"]) == 1:
                local[g["qubits"][0]] = self.step(g["name"], g.get("params"))
            else:
                raise ValueError("Unsupported two-qubit noise instruction : {}".format(g["name"]))
        value = 1.0
        for q in (0, 1):
            steps = []
            if q in pauli:
                steps.append(self.step(pauli[q], None))
            steps.append(local.get(q, ("id", None, None)))
            value *= self.chain(tuple(steps))
        return value

    def entry(
            self,
            entry : dict
    ) -> float:
        n_qubits = len(entry["gate_qubits"][0])
        term_value = self.single_qubit_term if n_qubits == 1 else self.two_qubit_term
        total = float(sum(entry["probabilities"]))
        if total <= 0:
            raise ValueError("Error entry {} on {} has no probability mass".format(entry.get("operations"), entry["gate_qubits"]))
        return sum(p * term_value(instruction) for p, instruction in zip(entry["probabilities"], entry["instructions"])) / total


class QubitLayoutOptimizer:
    def __init__(
            self,
            noise_model : dict,
            basis_gates : list[list[str]],
            interaction : str = "nearest_neighbour",
            error_cutoff : float | dict | None = None
    ) -> None:
        if not isinstance(interaction, str):
            raise TypeError("interaction must be of type string")
        if interaction not in INTERACTIONS:
            raise ValueError("interaction must be one of {}".format(INTERACTIONS))
        self.interaction = interaction
        self.error_cutoff = self._parse_error_cutoff(error_cutoff)
        self._read_noise_model(noise_model, basis_gates)
        self.node_weight = self._node_weights()
        self.edge_weight = self._edge_weights()
        self.excluded_qubits, self.excluded_couplers = self._apply_error_cutoff()
        self.adjacency = {
            q : set() for q in self.node_weight
        }
        for edge in self.edge_weight:
            a, b = tuple(edge)
            if a in self.adjacency and b in self.adjacency:
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)
        self.is_optimal = None

    def _read_noise_model(
            self,
            noise_model : dict,
            basis_gates : list[list[str]]
    ) -> None:
        single_gates, double_gates = set(basis_gates[0]), set(basis_gates[1])
        calculator = _ProcessFidelityCalculator()
        self.single_qubit_infidelity, self.readout_error = {}, {}
        self.two_qubit_infidelity = {
            g : {} for g in basis_gates[1]
        }
        for entry in noise_model["errors"]:
            operations = entry.get("operations", [])
            gate_qubits = entry.get("gate_qubits", [])
            if entry.get("type") == "roerror":
                for qubits in gate_qubits:
                    self.readout_error[qubits[0]] = readout_infidelity(entry["probabilities"])
                continue
            if entry.get("type") != "qerror" or not gate_qubits:
                continue
            if len(gate_qubits[0]) == 1:
                targets = [
                    op for op in operations if op in single_gates
                ]
                if not targets:
                    continue
                error = _average_log_infidelity(calculator.entry(entry), 2)
                for qubits in gate_qubits:
                    for op in targets:
                        self.single_qubit_infidelity.setdefault(qubits[0], {})[op] = error
            elif len(gate_qubits[0]) == 2 and operations and operations[0] in double_gates:
                error = _average_log_infidelity(calculator.entry(entry), 4)
                for qubits in gate_qubits:
                    self.two_qubit_infidelity[operations[0]][tuple(qubits)] = error

    def _node_weights(self) -> dict:
        log_f1 = {
            q : float(np.mean([log_fidelity(e) for e in per_gate.values()])) for q, per_gate in self.single_qubit_infidelity.items() if per_gate
        }
        qubits = set(self.single_qubit_infidelity) | set(self.readout_error)
        worst = min(log_f1.values()) if log_f1 else 0.0
        missing = sorted(qubits - set(log_f1))
        if missing:
            warnings.warn("No single qubit noise data for qubits {} found. Setting them to worst qubits".format(missing))
        return {
            q : log_f1.get(q, worst) + log_fidelity(self.readout_error.get(q, 0.0)) for q in qubits
        }

    def _edge_weights(self) -> dict:
        per_edge = {}
        for g, pairs in self.two_qubit_infidelity.items():
            for (q1, q2), error in pairs.items():
                per_edge.setdefault(frozenset((q1, q2)), {}).setdefault(g, []).append(log_fidelity(error))
        return {
            e : float(np.mean([np.mean(v) for v in per_gate.values()])) for e, per_gate in per_edge.items()
        }

    @staticmethod
    def _parse_error_cutoff(error_cutoff : float | dict | None) -> dict:
        if error_cutoff is None:
            return {}
        if isinstance(error_cutoff, (int, float)) and not isinstance(error_cutoff, bool):
            cutoff = {
                key : float(error_cutoff) for key in _CUTOFF_KEYS
            }
        elif isinstance(error_cutoff, dict):
            unknown_keys = set(error_cutoff) - set(_CUTOFF_KEYS)
            if unknown_keys:
                raise ValueError("Unknown keys found in error cutoff {}\nUse {} instead".format(sorted(unknown_keys), _CUTOFF_KEYS))
            cutoff = {
                key : float(value) for key, value in error_cutoff.items() if value is not None
            }
        else:
            raise TypeError("Unknown type for error_cutoff")
        if any(not 0.0 <= value <= 1.0 for value in cutoff.values()):
            raise ValueError("Cutoff values must be between 0 and 1")
        return cutoff

    def _apply_error_cutoff(self) -> tuple[list[int], list[tuple[int, int]]]:
        cutoff = self.error_cutoff
        excluded_qubits = set()
        if "single" in cutoff:
            excluded_qubits |= {
                q for q, per_gate in self.single_qubit_infidelity.items() if per_gate and max(per_gate.values()) > cutoff["single"]
            }
        if "readout" in cutoff:
            excluded_qubits |= {
                q for q, error in self.readout_error.items() if error > cutoff["readout"]
            }
        excluded_couplers = set()
        if "double" in cutoff:
            for pairs in self.two_qubit_infidelity.values():
                excluded_couplers |= {
                    frozenset(pair) for pair, error in pairs.items() if error > cutoff["double"]
                }
        for qubit in excluded_qubits:
            self.node_weight.pop(qubit, None)
        for coupler in excluded_couplers:
            self.edge_weight.pop(coupler, None)
        if excluded_qubits or excluded_couplers:
            warnings.warn("Following Qubits and couplers excluded due to high errors.\nExcluded Qubits : {}\nExcluded Couplers : {}".format(excluded_qubits, excluded_couplers))
        return sorted(excluded_qubits), sorted(tuple(sorted(e)) for e in excluded_couplers)

    def _induced_distances(
            self, 
            qubits : list[int]
    ) -> dict:
        members = set(qubits)
        dist = {}
        for source in qubits:
            seen = {source : 0}
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in self.adjacency[u]:
                    if v in members and v not in seen:
                        seen[v] = seen[u] + 1
                        queue.append(v)
            dist[source] = seen
        return dist

    def layout_cost(
        self,
        layout : list[int]
    ) -> tuple[int, float]:
        unavailable = sorted(set(layout) - set(self.adjacency))
        if unavailable:
            raise ValueError("Qubits {} are unavailable due to exclusion by high error rates or no data".format(unavailable))
        dist = self._induced_distances(layout)
        if len(dist[layout[0]]) != len(layout):
            raise ValueError("Selected qubits are not connected.")
        fidelity = sum(self.node_weight[q] for q in layout)
        members = set(layout)
        if self.interaction == INTERACTIONS[0]:
            hops = sum(dist[a][b] - 1 for a, b in zip(layout, layout[1:]))
            fidelity += sum(self.edge_weight[frozenset((a, b))] for a, b in zip(layout, layout[1:]) if frozenset((a, b)) in self.edge_weight)
        else:
            hops = sum(dist[a][b] - 1 for i, a in enumerate(layout) for b in layout[i + 1:])
            fidelity += sum(w for e, w in self.edge_weight.items() if e <= members)
        return hops, fidelity

    def best_layout(
            self,
            n : int,
            time_limit : float | None = 60.0,
            n_jobs : int = 1
    ) -> list[int]:
        if not isinstance(n_jobs, int):
            raise TypeError("n_jobs must be an integer")
        if not isinstance(n, int):
            raise TypeError("n needs to be an integer")
        if time_limit is None:
            time_limit = 60.0
        if not isinstance(time_limit, (int, float)):
            raise TypeError("time_limit must be a float or float-like")
        if n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be positive or -1.")
        if time_limit <= 0.0:
            raise ValueError("time_limit cannot be less than or equal to 0")
        if n < 1 or n > len(self.node_weight):
            raise ValueError("Requested qubits are out of bounds")
        if n == 1:
            self.is_optimal = True
            return [max(self.node_weight, key = lambda q: (self.node_weight[q], -q))]
        if self.interaction == INTERACTIONS[1]:
            return self._best_qubit_cluster(n, time_limit, n_jobs)
        if self.interaction == INTERACTIONS[0]:
            path = self._best_qubit_chain(n, time_limit)
            if path is not None:
                return path
            warnings.warn("No qubit chain satisfies the interaction, determining a cluster solution")
            layout = self._best_qubit_cluster(n, time_limit, n_jobs)
            self.is_optimal = False
            return layout

    def _best_qubit_chain(
            self,
            n : int,
            time_limit : float | None
    ) -> list[int] | int:
        adjacency = {
            q : sorted(nbs) for q, nbs in self.adjacency.items()
        }
        steps = [
            self.node_weight[b] + self.edge_weight[frozenset((a, b))] for a in adjacency for b in adjacency[a]
        ]
        max_steps = max(steps) if steps else 0.0
        best = {
            "score" : -math.inf,
            "path" : None
        }
        deadline = None if time_limit is None else time.monotonic() + time_limit
        state = {
            "expanded" : 0,
            "stopped" : False
        }

        def direct_path_score(
                path : list[int],
                on_path : set,
                score : float
        ) -> None:
            if state["stopped"]:
                return
            state["expanded"] += 1
            if deadline is not None and state["expanded"] % 4096 == 0 and time.monotonic() > deadline:
                state["stopped"] = True
                return
            if score + (n - len(path)) * max_steps < best["score"]:
                return
            if len(path) == n:
                if path[0] <= path[-1] and (score > best["score"]) or (score == best["score"] and path < best["path"]):
                        best["score"], best["path"] = score, list(path)
                return
            for nb in adjacency[path[-1]]:
                if nb not in on_path:
                    path.append(nb)
                    on_path.add(nb)
                    direct_path_score(path, on_path, score + self.node_weight[nb] + self.edge_weight[frozenset((path[-2], nb))])
                    path.pop()
                    on_path.discard(nb)

        for start in sorted(adjacency):
            direct_path_score([start], {start}, self.node_weight[start])
        self.is_optimal = not state["stopped"]
        if state["stopped"]:
            warnings.warn("Layout search stopped due to time limit")
        return best["path"]

    def _qubit_chain_order(
            self,
            qubit_cluster : set
    ) -> list[int]:
        start = min(qubit_cluster, key = lambda q: (len(self.adjacency[q] & qubit_cluster), q))
        order, stack, seen = [], [start], set()
        while stack:
            q = stack.pop()
            if q in seen:
                continue
            seen.add(q)
            order.append(q)
            stack.extend(sorted(self.adjacency[q] & qubit_cluster - seen, reverse = True))
        return order

    def _qubit_cluster_key(
            self,
            qubit_cluster : set
    ) -> tuple:
        ordered_cluster = sorted(qubit_cluster)
        hops, fidelity = self.layout_cost(ordered_cluster)
        return (hops, -fidelity, tuple(ordered_cluster))

    def _device_distances(self) -> tuple[list[int], dict, np.ndarray]:
        if not hasattr(self, "_device_distances_cache"):
            nodes = sorted(self.adjacency)
            index = {
                q : i for i, q in enumerate(nodes)
            }
            D = np.full((len(nodes), len(nodes)), len(nodes), dtype=np.int64)
            for q in nodes:
                for v, d in self._induced_distances_device(q).items():
                    D[index[q], index[v]] = d
            self._device_distances_cache = (nodes, index, D)
        return self._device_distances_cache

    def _induced_distances_device(
            self,
            source : int
    ) -> dict:
        seen = {
            source : 0
        }
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in self.adjacency[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    queue.append(v)
        return seen

    def _is_connected(
            self,
            members : set
    ) -> bool:
        if not members:
            return True
        start = next(iter(members))
        seen, stack = {start}, [start]
        while stack:
            for v in self.adjacency[stack.pop()] & members:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(members)

    def _local_search(
            self,
            key : tuple,
            max_checks : int = 300
    ) -> tuple:
        _, index, D = self._device_distances()
        members = set(key[2])
        while True:
            row_sum = D[[index[u] for u in members]].sum(axis=0)
            moves = []
            for u in members:
                rest = members - {u}
                if not self._is_connected(rest):
                    continue
                loss = self.node_weight[u] + sum(self.edge_weight[frozenset((u, m))] for m in self.adjacency[u] & rest)
                frontier = set().union(*(self.adjacency[m] for m in rest)) - members
                for w in frontier:
                    d_hops = int(row_sum[index[w]] - D[index[w], index[u]] - row_sum[index[u]])
                    if d_hops > 0:
                        continue
                    gain = self.node_weight[w] + sum(self.edge_weight[frozenset((w, m))] for m in self.adjacency[w] & rest)
                    if d_hops == 0 and gain <= loss:
                        continue
                    moves.append((d_hops, loss - gain, u, w))
            moves.sort()
            for _, _, u, w in moves[:max_checks]:
                candidate = self._qubit_cluster_key((members - {u}) | {w})
                if candidate < key:
                    key, members = candidate, set(candidate[2])
                    break
            else:
                return key

    def _greedy_qubit_clusters(
            self,
            n : int
    ) -> list[tuple]:
        nodes, index, D = self._device_distances()
        keys = set()
        for start in nodes:
            chosen, dist_sum = [start], D[index[start]].astype(np.int64).copy()
            frontier = set(self.adjacency[start])
            while len(chosen) < n and frontier:
                nxt = min(frontier, key = lambda q : (dist_sum[index[q]], -self.node_weight[q], q))
                chosen.append(nxt)
                dist_sum += D[index[nxt]]
                frontier |= self.adjacency[nxt]
                frontier -= set(chosen)
            if len(chosen) == n:
                keys.add(self._qubit_cluster_key(chosen))
        return sorted(keys)

    def _placements(
            self,
            qubit_cluster : tuple,
            limit : int = 200_000
    ) -> set:
        members = set(qubit_cluster)
        pattern = {
            u : self.adjacency[u] & members for u in members
        }
        root = max(members, key = lambda u: (len(pattern[u]), -u))
        order, seen = [root], {root}
        for u in order:
            for v in sorted(pattern[u]):
                if v not in seen:
                    seen.add(v)
                    order.append(v)
        position = {
            u : i for i, u in enumerate(order)
        }
        earlier = [
            [v for v in pattern[u] if position[v] < position[u]] for u in order
        ]
        degree = [len(pattern[u]) for u in order]
        images, found, state = [None] * len(order), set(), {"steps" : 0}

        def extend_set(
            i : int,
            used : set
        ) -> None:
            if state["steps"] > limit:
                return
            state["steps"] += 1
            if i == len(order):
                found.add(frozenset(images))
                return
            if i == 0:
                candidates = sorted(self.adjacency)
            else:
                anchor = images[position[earlier[i][0]]]
                candidates = sorted(self.adjacency[anchor] - used)
            for c in candidates:
                if len(self.adjacency[c]) < degree[i]:
                    continue
                if all(images[position[v]] in self.adjacency[c] for v in earlier[i]):
                    images[i] = c
                    used.add(c)
                    extend_set(i + 1, used)
                    used.discard(c)
            images[i] = None

        extend_set(0, set())
        return found

    def _refine(
            self,
            key : tuple
    ) -> tuple:
        best = key
        for placement in self._placements(key[2]):
            if self._is_connected(set(placement)):
                best = min(best, self._qubit_cluster_key(placement))
        return self._local_search(best)

    def _heuristic_qubit_cluster(
            self,
            n : int,
            pool_size : int = 16
    ) -> tuple:
        pool = self._greedy_qubit_clusters(n)
        if not pool:
            raise ValueError("There are no connected components to qubit {}".format(n))
        improved = sorted({self._local_search(key) for key in pool[:pool_size]})
        best = improved[0]
        for key in improved:
            if key[0] == best[0]:
                best = min(best, self._refine(key))
        return best

    def _qubit_cluster_context(
            self,
            n : int
    ) -> int:
        nodes, index, D = self._device_distances()
        size = len(nodes)
        nearest = np.sort(D + np.diag(np.full(size, 10**9)), axis=1)
        prefix = np.cumsum(nearest, axis=1)
        w_min = [0] * (n + 1)
        for r in range(2, n + 1):
            w_min[r] = int(math.ceil(np.sort(prefix[:, r - 2])[:r].sum() / 2))
        return {
            "nodes" : nodes,
            "index" : index,
            "D" : D,
            "w_min" : w_min,
            "node_w" : np.array([self.node_weight[q] for q in nodes]),
            "best_edge" : max(self.edge_weight.values()) if self.edge_weight else 0.0
        }

    def _search_tasks(
            self, 
            n : int
    ) -> list[tuple[int, int]]:
        nodes, index, D = self._device_distances()
        tasks = []
        for root in nodes:
            extension = sorted((u for u in self.adjacency[root] if u > root), key = lambda q : (D[index[root], index[q]], q))
            tasks.extend((root, i) for i in range(len(extension)))
        return tasks

    def _run_search_tasks(
            self,
            n : int,
            tasks : list,
            incumbent : tuple,
            deadline : float | None
    ) -> tuple:
        context = self._qubit_cluster_context(n)
        index, D, w_min = context["index"], context["D"], context["w_min"]
        node_w, best_edge = context["node_w"], context["best_edge"]
        n_pairs = n * (n - 1) // 2
        best = {
            "key" : incumbent
        }
        state = {
            "expanded" : 0,
            "stopped" : False
        }

        def prune(
                members : set,
                dist_sum : np.ndarray,
                pair_sum : float,
                log_fid : float
        ) -> bool:
            r = n - len(members)
            outside = dist_sum.astype(np.float64)
            outside[[index[q] for q in members]] = np.inf
            cross = np.partition(outside, r - 1)[:r].sum()
            hops_lb = pair_sum + cross + w_min[r] - n_pairs
            fid_ub = log_fid + np.sort(node_w[np.isfinite(outside)])[-r:].sum() + r * best_edge
            return hops_lb > best["key"][0] or (hops_lb == best["key"][0] and -fid_ub > best["key"][1] + 1e-12)

        def extend(
                members : set,
                extension : list | set,
                root : set,
                neighbourhood : set,
                dist_sum : np.ndarray,
                pair_sum : float,
                log_fid : float
        ) -> None:
            if state["stopped"]:
                return
            state["expanded"] += 1
            if deadline is not None and state["expanded"] % 1024 == 0 and time.time() > deadline:
                state["stopped"] = True
                return
            if len(members) == n:
                key = self._qubit_cluster_key(members)
                if key < best["key"]:
                    best["key"] = key
                return
            if prune(members, dist_sum, pair_sum, log_fid):
                return
            extension = sorted(extension, key = lambda q : (dist_sum[index[q]], q))
            while extension:
                w = extension.pop(0)
                new_extension = set(extension) | {u for u in self.adjacency[w] if u > root and u not in neighbourhood}
                new_log_fid = log_fid + self.node_weight[w] + sum(
                    self.edge_weight[frozenset((w, m))] for m in members if m in self.adjacency[w]
                )
                extend(
                    members + [w],
                    new_extension,
                    root,
                    neighbourhood | self.adjacency[w],
                    dist_sum + D[index[w]],
                    pair_sum + int(dist_sum[index[w]]),
                    new_log_fid
                )

        for root, i in tasks:
            first = sorted((u for u in self.adjacency[root] if u > root), key = lambda q : (D[index[root], index[q]], q))
            w = first[i]
            neighbourhood = self.adjacency[root] | {root}
            extension = set(first[i + 1:]) | {u for u in self.adjacency[w] if u > root and u not in neighbourhood}
            extend(
                [root, w],
                extension,
                root, 
                neighbourhood | self.adjacency[w] | {w},
                D[index[root]] + D[index[w]],
                int(D[index[root], index[w]]),
                self.node_weight[root] + self.node_weight[w] + self.edge_weight[frozenset((root, w))]
            )
            if state["stopped"]:
                break
        return best["key"], not state["stopped"]

    def _best_qubit_cluster(
            self,
            n : int, 
            time_limit : float | None,
            n_jobs : int
    ) -> list[int]:
        best = self._heuristic_qubit_cluster(n)
        self.is_optimal = False
        deadline = None if time_limit is None else time.time() + time_limit
        tasks = self._search_tasks(n)
        if n_jobs == 1:
            results = [
                self._run_search_tasks(n, tasks, best, deadline)
            ]
        else:
            if n_jobs == -1:
                n_jobs = len(os.sched_getaffinity(0))
            chunks = [tasks[k::n_jobs * 4] for k in range(n_jobs * 4)]
            results = Parallel(n_jobs = n_jobs)(delayed(
                self._run_search_tasks)(n, chunk, best, deadline) for chunk in chunks if chunk
            )
        searched = min([best] + [key for key, _ in results])
        self.is_optimal = all(done for _, done in results)
        if not self.is_optimal and searched < best:
            searched = self._refine(searched)
        best = min(best, searched)
        if not self.is_optimal:
            warnings.warn(f"Search Limit Reached, layout optimality unproven")
        return self._qubit_cluster_order(list(best[2]))

    def _qubit_cluster_order(
            self,
            qubit_cluster : list[int]
    ) -> list[int]:
        return self._qubit_chain_order(set(qubit_cluster))