"""Deploy compression actions into the running language model"""

from __future__ import annotations

import copy

import numpy as np


def effective_bias_of(parent, eps):
    """the unnormalized bias realizing the parent's BN parameterization"""
    scale = parent["gamma"] / np.sqrt(parent["sigma"] ** 2 + eps)
    return float(parent["beta"] - scale * parent["mu"])


def clone_model(model):
    return copy.deepcopy(model)


def prune_unit(model, i):
    m = clone_model(model)
    keep = [k for k in range(m.W1.shape[1]) if k != i]
    m.W1 = m.W1[:, keep]
    m.b1 = m.b1[keep]
    m.W2 = m.W2[keep, :]
    return m


def merge_units(model, i, j, parent, eps):
    m = clone_model(model)
    scale = parent["gamma"] / np.sqrt(parent["sigma"] ** 2 + eps)
    m.W1 = m.W1.copy(); m.b1 = m.b1.copy(); m.W2 = m.W2.copy()
    m.W1[:, i] = scale * parent["w_raw"]     
    m.b1[i] = effective_bias_of(parent, eps)
    m.W2[i, :] = parent["w_out"]
    keep = [k for k in range(m.W1.shape[1]) if k != j]
    m.W1 = m.W1[:, keep]
    m.b1 = m.b1[keep]
    m.W2 = m.W2[keep, :]
    return m
