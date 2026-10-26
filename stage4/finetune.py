"""recovery blocks: low-rate fine-tuning along the eviction trajectory"""

from __future__ import annotations

import numpy as np


def reset_adam(model):
    # unit surgery changes parameter shapes, so stale moments would crash fit()
    model._adam = {p: [np.zeros_like(getattr(model, p)),
                       np.zeros_like(getattr(model, p))]
                   for p in model._params}


def ft_block(model, train_ids, steps, lr, seed, batch=256):
    reset_adam(model)
    model.fit(train_ids, steps=steps, batch=batch, lr=lr, seed=seed,
              log_every=10**9)
    return model


def checkpoint_densities(start=0.95, stop=0.35, step=0.05):
    cps = [round(d, 2) for d in np.arange(start, stop + step / 2, -step)]
    if cps[-1] != round(stop, 2):
        cps.append(round(stop, 2))
    return cps


class FTLog:
    """one row per recovery block, plus total-budget accounting"""

    def __init__(self):
        self.rows = []

    def add(self, arm, mode, density, steps, lr, val_before, val_after):
        self.rows.append(dict(arm=arm, mode=mode, density=density,
                              steps=steps, lr=lr, val_before=val_before,
                              val_after=val_after))

    def total_steps(self, arm, mode):
        return sum(r["steps"] for r in self.rows
                   if r["arm"] == arm and r["mode"] == mode)
