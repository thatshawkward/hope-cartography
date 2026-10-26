"""replay recorded eviction sequences into a live model, with recovery hooks"""

from __future__ import annotations

from stage3.writeback import merge_units, prune_unit


def replay(model, events, parents, h0, eps, val, checkpoints=(), hook=None):
    """Apply a recorded action sequence verbatim.

    Merges reuse the parents synthesized during the base run, so the plan is
    frozen even when a hook has since changed the weights. hook(model, density,
    step) -> model fires once per crossed checkpoint density; each firing adds
    a second val point at the same density, making recovery visible in the
    curve. Returns (final_model, val_curve).
    """
    ids = list(range(h0))
    curve = [(1.0, model.loss(*val))]
    cps = sorted(checkpoints, reverse=True)
    for e in events:
        if e.action == "prune":
            (u,) = e.removed_ids
            model = prune_unit(model, ids.index(u))
            ids.remove(u)
        else:
            u_i, u_j = e.removed_ids
            i, j = ids.index(u_i), ids.index(u_j)
            model = merge_units(model, i, j, parents[e.parent_id], eps)
            ids[i] = e.parent_id
            ids.pop(j)
        density = len(ids) / h0
        curve.append((density, model.loss(*val)))
        while cps and density <= cps[0] + 1e-12:
            cps.pop(0)
            if hook is not None:
                model = hook(model, density, e.step)
                curve.append((density, model.loss(*val)))
    return model, curve


def replay_prune_order(model, order, h0, val, stop_density,
                       checkpoints=(), hook=None):
    """static_prune_curve with recovery hooks, for the baseline orders"""
    alive = list(range(h0))
    curve = [(1.0, model.loss(*val))]
    cps = sorted(checkpoints, reverse=True)
    step = 0
    for orig in order:
        if len(alive) / h0 <= stop_density:
            break
        model = prune_unit(model, alive.index(orig))
        alive.remove(orig)
        step += 1
        density = len(alive) / h0
        curve.append((density, model.loss(*val)))
        while cps and density <= cps[0] + 1e-12:
            cps.pop(0)
            if hook is not None:
                model = hook(model, density, step)
                curve.append((density, model.loss(*val)))
    return model, curve
