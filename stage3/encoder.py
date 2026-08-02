"""greedy progressive encoding with the ledger as product"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minihope.merge import merge_pair, prune_cost
from stage2.experiments import rho_hat_matrix

from .writeback import merge_units, prune_unit


@dataclass
class MapEvent:
    step: int
    action: str                 
    removed_ids: tuple          
    removed_labels: tuple
    removed_exkurt: tuple
    removed_caps: tuple
    parent_id: object
    rho: object
    cost: float
    n_after: int
    density: float
    val_loss: float


@dataclass
class MapResult:
    events: list = field(default_factory=list)
    lineage: dict = field(default_factory=dict)   #parent id >> (id_i, id_j)
    surviving_ids: np.ndarray = None
    final_model: object = None
    val_curve: list = field(default_factory=list)


def encode_and_map(model, layer, unit_labels, unit_exkurt, val,
                   probe_ctx=None, mode="paper", target_density=0.35,
                   rho_min=0.10, max_pairs=300, min_units=8, verbose_every=40):
    assert mode in ("paper", "empirical")
    h0 = layer.n_neurons
    ids = np.arange(h0)
    next_id = h0
    labels = dict(enumerate(unit_labels))
    exkurt = dict(enumerate(np.asarray(unit_exkurt, dtype=float)))
    res = MapResult()
    res.val_curve.append((1.0, model.loss(*val)))
    step = 0

    while layer.n_neurons / h0 > target_density and layer.n_neurons > min_units:
        n = layer.n_neurons
        assert model.W1.shape[1] == n, "model/layer desynchronized"
        caps = layer.capacities()
        e_a = float(caps.sum())

        if mode == "empirical":
            hp = model.preactivations(probe_ctx)
            rho_mx = np.corrcoef(hp.T)
            np.fill_diagonal(rho_mx, 1.0)
        else:
            rho_mx = rho_hat_matrix(layer)

        #price all prunes and the top merge candidates
        j_prune = n * caps / np.maximum(e_a - caps, 1e-12)
        best = dict(action="prune", i=int(np.argmin(j_prune)),
                    cost=float(j_prune.min()), result=None, rho=None)
        iu = np.triu_indices(n, k=1)
        cand = np.where(rho_mx[iu] >= rho_min)[0]
        if cand.size > max_pairs:
            cand = cand[np.argsort(rho_mx[iu][cand])[::-1][:max_pairs]]
        for c in cand:
            i, j = int(iu[0][c]), int(iu[1][c])
            r = merge_pair(layer, i, j, layer_capacity=e_a,
                           rho=float(rho_mx[i, j]))
            if np.isfinite(r.cost) and r.cost < best["cost"]:
                best = dict(action="merge", i=i, j=j, cost=float(r.cost),
                            result=r, rho=float(rho_mx[i, j]))

        #execute with writeback
        if best["action"] == "prune":
            i = best["i"]
            removed = (int(ids[i]),)
            event_caps = (float(caps[i]),)
            model = prune_unit(model, i)
            layer = layer.subset([k for k in range(n) if k != i])
            ids = np.delete(ids, i)
            parent_id = None
        else:
            i, j, r = best["i"], best["j"], best["result"]
            removed = (int(ids[i]), int(ids[j]))
            event_caps = (float(caps[i]), float(caps[j]))
            model = merge_units(model, i, j, r.parent, layer.eps)
            layer = layer.with_neuron_replaced(i, r.parent)
            layer = layer.subset([k for k in range(n) if k != j])
            parent_id = next_id
            res.lineage[parent_id] = removed
            labels[parent_id] = "parent"
            exkurt[parent_id] = float("nan")
            new_ids = ids.copy(); new_ids[i] = parent_id
            ids = np.delete(new_ids, j)
            next_id += 1

        step += 1
        vloss = model.loss(*val)
        density = layer.n_neurons / h0
        res.val_curve.append((density, vloss))
        res.events.append(MapEvent(
            step=step, action=best["action"], removed_ids=removed,
            removed_labels=tuple(labels[u] for u in removed),
            removed_exkurt=tuple(exkurt[u] for u in removed),
            removed_caps=event_caps, parent_id=parent_id, rho=best["rho"],
            cost=float(best["cost"]), n_after=layer.n_neurons, density=density,
            val_loss=vloss))
        if verbose_every and step % verbose_every == 0:
            print(f"  [{mode}] step {step:>4}  N {layer.n_neurons:>4}  "
                  f"density {density:.3f}  val {vloss:.4f}", flush=True)

    res.surviving_ids = ids
    res.final_model = model
    return res


def static_prune_curve(model, order, val, stop_density=0.35):
    """Validation loss along a fixed removal order (for baselines)."""
    h0 = model.W1.shape[1]
    curve = [(1.0, model.loss(*val))]
    m = model
    alive = list(range(h0))
    for orig in order:
        if len(alive) / h0 <= stop_density:
            break
        pos = alive.index(orig)
        m = prune_unit(m, pos)
        alive.pop(pos)
        curve.append((len(alive) / h0, m.loss(*val)))
    return curve
