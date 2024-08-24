import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def mpjpe(pose1, pose2):
    """Calculate Mean Per Joint Position Error (MPJPE) between two poses."""
    return np.mean(np.linalg.norm(pose1 - pose2, axis=1))

def auto_correlation(poses, tau, s):
    """Calculate auto-correlation of the pose signal."""
    N = len(poses)
    return np.mean([mpjpe(poses[t], poses[t+tau]) for t in range(s, N-s-tau)])

def affinity(pose1, pose2):
    """Calculate affinity between two poses."""
    return -mpjpe(pose1, pose2)

def uniform_sample(interval, poses, nS):
    """Uniformly sample nS frames from the given interval."""
    start, end = interval
    if start == end:
        return [poses[int(start)]] * nS
    t = np.linspace(start, end, nS)
    interpolator = interp1d(np.arange(len(poses)), poses, axis=0, kind='linear')
    return interpolator(t)

def seq_affinity(interval1, interval2, poses, nS):
    """Calculate sequence affinity between two intervals."""
    samples1 = uniform_sample(interval1, poses, nS)
    samples2 = uniform_sample(interval2, poses, nS)
    return np.mean([affinity(s1, s2) for s1, s2 in zip(samples1, samples2)])

def avg_affinity(intervals, poses, nS):
    """Calculate average affinity for all interval pairs."""
    k = len(intervals)
    affinities = [seq_affinity(intervals[i], intervals[j], poses, nS)
                  for i in range(k) for j in range(i+1, k)]
    return np.mean(affinities)

def initialize_segmentation(poses, kmin):
    """Initialize segmentation assuming fixed period."""
    N = len(poses)
    best_tau, best_s, best_corr = None, None, -np.inf
    
    for s in range(N // 4):  # Adjust range as needed
        for tau in range(1, (N - 2*s) // kmin):
            corr = auto_correlation(poses, tau, s)
            if corr > best_corr:
                best_tau, best_s, best_corr = tau, s, corr

    best_start, best_avg = None, -np.inf
    for start in range(best_s, N - best_s - kmin*best_tau):
        intervals = [(start + i*best_tau, start + (i+1)*best_tau) for i in range(kmin)]
        avg = avg_affinity(intervals, poses, nS=10)  # nS can be adjusted
        if avg > best_avg:
            best_start, best_avg = start, avg

    return [(best_start + i*best_tau, best_start + (i+1)*best_tau) for i in range(kmin)]

def optimize_segmentation(initial_intervals, poses, kmin):
    """Optimize segmentation using constrained continuous optimization."""
    N = len(poses)
    nS = 10  # Can be adjusted

    def objective(x):
        intervals = [(x[i], x[i+1]) for i in range(kmin)]
        return -avg_affinity(intervals, poses, nS)

    def constraints(x):
        return np.array([x[i+1] - x[i] - 1 for i in range(kmin)])  # Ensure intervals are at least 1 frame long

    x0 = [interval[0] for interval in initial_intervals] + [initial_intervals[-1][1]]
    bounds = [(0, N-1) for _ in range(kmin+1)]
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraints})
    
    optimized_intervals = [(result.x[i], result.x[i+1]) for i in range(kmin)]
    return optimized_intervals

def segment_repetitions(poses, kmin):
    """Segment repetitions of 2D human poses."""
    initial_intervals = initialize_segmentation(poses, kmin)
    optimized_intervals = optimize_segmentation(initial_intervals, poses, kmin)
    return optimized_intervals
