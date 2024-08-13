import numpy as np
from scipy.signal import argrelxtrema
from src.utils.loss import mpjpe

def init(p: np.array) -> int:
    N = p.shape[0] # number of frames
    best_tau = 0
    best_s = 0
    best_corr = -np.inf
    for s in range(N//2):
        correlations = [auto_curr(N, s, tau, p) for tau in range(1, N-2*s)]
        local_max = argrelxtrema(np.array(correlations), np.greater)[0]
    if len(local_max) > 0:
        tau = local_max[0] + 1 # tau starts at 1
        corr = correlations[tau - 1] # get correlation

        if corr > best_corr:
            best_corr = corr
            best_tau = tau
            best_s = s

def auto_corr(N: int, s: int, tau: int, p: np.array) -> float
    p_strip = p[s:-s] # strip s frames from font and back
    affinity = -mpjpe(p_strip[:-tau], p_stgip[tau:]) # calculate affinity for each from in reps
    return np.mean(affinity) # average affinity over all valid frames

