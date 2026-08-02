"""the greedy progressive encoding loop in single layer form """

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .merge import merge_pair, prune_cost


@dataclass
class LedgerRow:
    step: int
    action: str         
    targets: tuple       # neuron ids at the time of the action
    cost: float          # J
    dp: int              # static parameter yield Delta P
    dr: float            # distortion rate J/Delta P
    n_after: int
    capacity_after: float
    detail: str = ""


def greedy_encode(layer, target_density=0.5, max_steps=None, min_neurons=2):
    n0 = layer.n_neurons
    dp = layer.n_in + layer.n_out + 4         
    ledger = []
    step = 0
    while layer.n_neurons / n0 > target_density and layer.n_neurons > min_neurons:
        if max_steps is not None and step >= max_steps:
            break
        caps = layer.capacities()
        e_a = float(caps.sum())
        n = layer.n_neurons

        best = None
        for i in range(n):
            cost = prune_cost(caps[i], e_a, n)
            if best is None or cost / dp < best["dr"]:
                best = dict(action="prune", targets=(i,), cost=cost,
                            dr=cost / dp, result=None)
        for i in range(n):
            for j in range(i + 1, n):
                res = merge_pair(layer, i, j, layer_capacity=e_a)
                if np.isfinite(res.cost) and res.cost / dp < best["dr"]:
                    best = dict(action="merge", targets=(i, j), cost=res.cost,
                                dr=res.cost / dp, result=res)

        if best["action"] == "prune":
            (i,) = best["targets"]
            keep = [k for k in range(n) if k != i]
            layer = layer.subset(keep)
            detail = f"capacity {caps[i]:.4f}"
        else:
            i, j = best["targets"]
            res = best["result"]
            layer = layer.with_neuron_replaced(i, res.parent)
            keep = [k for k in range(layer.n_neurons) if k != j]
            layer = layer.subset(keep)
            detail = (f"rho_hat {res.rho_hat:+.3f}, D {res.distortion:.4f}, "
                      f"||f_p|| {res.s_star:.4f}")

        step += 1
        ledger.append(LedgerRow(step=step, action=best["action"],
                                targets=best["targets"], cost=best["cost"],
                                dp=dp, dr=best["dr"],
                                n_after=layer.n_neurons,
                                capacity_after=float(layer.capacities().sum()),
                                detail=detail))
    return layer, ledger


def format_ledger(ledger):
    lines = [f"{'step':>4}  {'action':<6} {'targets':<10} {'J':>9} "
             f"{'DR':>9} {'N':>4} {'E':>9}  detail"]
    for row in ledger:
        lines.append(f"{row.step:>4}  {row.action:<6} {str(row.targets):<10} "
                     f"{row.cost:>9.4f} {row.dr:>9.5f} {row.n_after:>4} "
                     f"{row.capacity_after:>9.4f}  {row.detail}")
    return "\n".join(lines)
