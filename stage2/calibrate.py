from __future__ import annotations
import numpy as np
from minihope.layer import HopeLayer


def calibrated_layer(model, calib_ctx, eps=1e-8):
    h_pre = model.preactivations(calib_ctx)        
    mean_y = h_pre.mean(axis=0)
    std_y = h_pre.std(axis=0)
    mu = mean_y - model.b1                           
    sigma = std_y                                     
    return HopeLayer(w_raw=model.W1.T, w_out=model.W2,
                     gamma=sigma, beta=mean_y, mu=mu, sigma=sigma, eps=eps)
