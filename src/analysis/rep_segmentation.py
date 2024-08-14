import numpy as np
from typing import Tuple
from scipy.signal import argrelxtrema
from src.utils.loss import mpjpe

def init(p: np.array) -> Tuple(int, int):
    '''
    Initialize estimate of rep segments.

    This assumes a constant tau value for rep length and a single s value to strip noise from font and back

    Parameters:
        p: numpy array
        Poses to split into repetitions. Pixel coord 2d keypoints

    Returns:
        Tuple(int, int)
        Initial estimates tau^* and s^*  '''
    N = p.shape[0] # number of frames
    best_tau = 0
    best_s = 0
    best_corr = -np.inf
    for s in range(N//2):
        correlations = [auto_corr(N, s, tau, p) for tau in range(1, N-2*s)]
        local_max = argrelxtrema(np.array(correlations), np.greater)[0]

        if len(local_max) > 0:
            tau = local_max[0] + 1 # tau starts at 1
            corr = correlations[tau - 1] # get correlation

            if corr > best_corr:
                best_corr = corr
                best_tau = tau
                best_s = s

    return best_tau, best_s

def auto_corr(N: int, s: int, tau: int, p: np.array) -> float:
    '''
    Auto-correlation of rep segments.

    This is the auto-correlation based on parameters tau and s on poses p.

    Parameters:
        N: int
        Number of total frames

        s: int
        Number of frames to strip from front and back

        tau: int
        Length of single repetition. Measuered in frames

        p: np.array
        Poses to split. 2d keypoints. (N, J, 2)

    Returns:
        float
        Auto-correlation
    '''
    p_strip = p[s:-s] # strip s frames from font and back
    affinity = -mpjpe(p_strip[:-tau], p_strip[tau:]) # calculate affinity for each from in reps
    return np.mean(affinity) # average affinity over all valid frames

def avg_aff(tau: int, s: int, p: np.array) -> float:
    best_corr = -np.inf
    best_t_start = 0
    num_reps = len(p) - 2 * s // tau
    for t_start in range(len(p) - 2 * s):
        affinity = 0
        for i in range(num_reps):
            for i in range(num_reps):
                affinity += seq_aff(t_start, i, j, tau, p)
        if affinity > best_corr:
            best_corr = affinity
            best_t_start = t_start

    return best_t_start

def seq_aff(t_start: int, i: int, j: int, tau: int, p: np.array) -> float:
    t_i = t_start + tau * i
    t_j = t_start + tau * j

    return np.mean(mpjpe(p[t_i: t_i+tau], p[t_j: t_j+tau]))


