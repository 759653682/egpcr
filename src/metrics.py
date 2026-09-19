import numpy as np
from sklearn.neighbors import KernelDensity

from .config import BAYESIAN_EPSILON


def spekde(stat_tr, bw, q, seed):
    stat_tr_cleaned = stat_tr[
        ~np.isnan(stat_tr) & ~np.isinf(stat_tr)
    ].reshape(-1, 1)

    n = stat_tr_cleaned.shape[0]
    if n == 0:
        return np.nan

    min_val, max_val = np.min(stat_tr_cleaned), np.max(stat_tr_cleaned)
    if n < 5 or bw <= 0 or abs(min_val - max_val) < BAYESIAN_EPSILON:
        return np.percentile(stat_tr_cleaned, q * 100)

    try:
        kde = KernelDensity(bandwidth=bw, kernel="gaussian").fit(stat_tr_cleaned)
        samples_from_kde = kde.sample(10000, random_state=seed)
        return np.percentile(samples_from_kde, q * 100)
    except Exception:
        return np.percentile(stat_tr_cleaned, q * 100)


def calculate_far_fdr(alarm_array, num_normal_samples, num_fault_samples):
    num_normal_to_check = min(len(alarm_array), num_normal_samples)
    far = (
        np.sum(alarm_array[:num_normal_to_check]) / num_normal_to_check
        if num_normal_to_check > 0
        else 0.0
    )

    fault_alarms_array = alarm_array[num_normal_to_check:]
    fdr = (
        np.sum(fault_alarms_array) / len(fault_alarms_array)
        if len(fault_alarms_array) > 0
        else 0.0
    )
    return far, fdr
