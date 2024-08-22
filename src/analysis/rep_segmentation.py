import math
import numpy as np
from typing import Tuple, List
from scipy.signal import argrelxtrema
from src.utils.loss import mpjpe

def init(p: np.ndarray) -> Tuple[int, int]:
    '''
    Initialize estimate of rep segments.

    This assumes a constant tau value for rep length and a single s value to strip noise from font and back

    Parameters:
        p: numpy array
        Poses to split into repetitions. Pixel coord 2d keypoints

    Returns:
        Tuple(int, int)
        Initial estimates tau^* and s^*
    '''
    N = p.shape[0] # number of frames
    best_tau = 0
    best_s = 0
    best_corr = -np.inf
    for s in range(N//2):
        correlations = [auto_corr(s, tau, p) for tau in range(1, N-2*s)]
        local_max = argrelxtrema(np.array(correlations), np.greater)[0]

        if len(local_max) > 0:
            tau = local_max[0] + 1 # tau starts at 1
            corr = correlations[tau - 1] # get correlation

            if corr > best_corr:
                best_corr = corr
                best_tau = tau
                best_s = s

    return best_tau, best_s

def find_start(k_min: int, s: int, tau: int, p: np.ndarray) -> int:
    '''
    Find start frame of the series of rep. t_start

    This optimzes the correlation parametrized by t_start

    Parameters:
        k_min: int
        Minimum number of repetitions

        s: int
        Initial approximation of start and end noise

        tau: int
        Initial approximation of length of repetitions

        p: numpy array
        Poses. 2d keypoints. (N, J, 2)

    Returns:
        int
        optimized t_start value
    '''
    best_corr = -np.inf # top correlation value
    best_t_start = s
    for t_start in range(len(p) - 2 * s):
        corr = avg_aff(k_min, t_start, tau, p) # average affintity across all k_min reps

        if corr > best_corr:
            best_corr = corr
            best_t_start = t_start

    return best_t_start


def auto_corr(s: int, tau: int, p: np.ndarray) -> float:
    '''
    Auto-correlation of rep segments.

    This is the auto-correlation based on parameters tau and s on poses p.

    Parameters:
        s: int
        Number of frames to strip from front and back

        tau: int
        Length of single repetition. Measuered in frames

        p: np.ndarray
        Poses to split. 2d keypoints. (N, J, 2)

    Returns:
        float
        Auto-correlation
    '''
    p_strip = p[s:-s] # strip s frames from font and back
    affinity = -mpjpe(p_strip[:-tau], p_strip[tau:]) # calculate affinity for each from in reps
    return np.mean(affinity) # average affinity over all valid frames

def avg_aff(k_min: int, t_start: int, tau: int, p: np.ndarray) -> float:
    '''
    Compute average affinity of reps based on t start.

    Parameters:
        k_min: int
        Minimum number of reps. k >= k_min

        t_start: int
        Frame number where reps start

        tau: int
        Number of frames in each rep

        p: np.ndarray
        Poses. 2d keypoints (N, J, 2)

    Returns:
        float
        average affinity score
    '''
    affinity = 0.0
    for i in range(k_min):
        for j in range(k_min):
            affinity += seq_aff(t_start, i, j, tau, p) # affinity between reps i and j

    return affinity / k_min ** 2

def seq_aff(t_start: int, i: int, j: int, tau: int, p: np.ndarray) -> float:
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

        p: np.ndarray
        Poses. 2d keypoints (N, J, 2)

    Returns:
        float
        affinity between reps i and j
    '''
    t_i = t_start + tau * i
    t_j = t_start + tau * j

    return np.mean(mpjpe(p[t_i: t_i+tau], p[t_j: t_j+tau]))

def uniform_sample(T_i: List[int], n_s: int, p: np.ndarray):
    result = []
    step = (len(T_i) - 1) / (n_s - 1)
    x = T_i[0]
    while x <= T_i[-1]:
        p_hat = p[math.floor(x)] * (1 - (x - math.floor(x))) + p[math.ceil(x)] * (x - math.floor(x))
        result.append(p_hat)
        x += step
    return result
