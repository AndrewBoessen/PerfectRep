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

def avg_aff(k_min: int, t_start: int, tau: int, p: np.array) -> float:
    '''
    Compute average affinity of reps based on t start.

    Parameters:
        k_min: int
        Minimum number of reps. k >= k_min

        t_start: int
        Frame number where reps start

        tau: int
        Number of frames in each rep

        p: np.array
        Poses. 2d keypoints (N, J, 2)

    Returns:
        float
        average affinity score
    '''
    affinity = 0
    for i in range(k_min):
        for j in range(k_min):
            affinity += seq_aff(t_start, i, j, tau, p)

    return affinity / k_min ** 2

def seq_aff(t_start: int, i: int, j: int, tau: int, p: np.array) -> float:
    '''
    Compute affinity between two repetitions.

    This uses mpjpe to calcualte correlation between reps i and j.

    Parameters:
        t_start: int
        Frame number of first rep

        i: int
        First repetition number

        j: int
        Second repetition number

        tau: int
        Number of frames in a single rep

        p: np.array
        Poses. 2d keypoints (N, J, 2)

    Returns:
        float
        affinity between reps i and j
    '''
    t_i = t_start + tau * i
    t_j = t_start + tau * j

    return np.mean(mpjpe(p[t_i: t_i+tau], p[t_j: t_j+tau]))


