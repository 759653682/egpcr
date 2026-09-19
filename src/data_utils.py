import os

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_train_data(data_dir, train_filename, cols_to_delete):
    train_file_path = os.path.join(data_dir, train_filename)
    raw_data = np.loadtxt(train_file_path)

    if cols_to_delete:
        valid_cols = [c for c in cols_to_delete if c < raw_data.shape[1]]
        raw_data = np.delete(raw_data, valid_cols, axis=1)

    scaler = StandardScaler().fit(raw_data)
    return raw_data, scaler, raw_data.shape[1]


def inject_multiplicative_noise(data, var_indices, levels, seed):
    rng = np.random.default_rng(seed)
    noisy_data = np.copy(data)

    if not var_indices or not levels:
        return noisy_data

    for var_idx, level in zip(var_indices, levels):
        if 0 <= var_idx < data.shape[1]:
            true_values = data[:, var_idx]
            random_multipliers = level * (2 * rng.random(size=true_values.shape) - 1)
            noise = true_values * random_multipliers
            noisy_data[:, var_idx] += noise

    return noisy_data


def prepare_global_windows_padded(data_2d, window_size):
    num_samples_orig, num_features = data_2d.shape
    padding_size = window_size - 1

    if padding_size > 0:
        first_sample = data_2d[0, :]
        padding = np.tile(first_sample, (padding_size, 1))
        data_2d_padded = np.concatenate([padding, data_2d], axis=0)
    else:
        data_2d_padded = data_2d

    shape = (num_samples_orig, window_size, num_features)
    strides = (
        data_2d_padded.strides[0],
        data_2d_padded.strides[0],
        data_2d_padded.strides[1],
    )
    return np.lib.stride_tricks.as_strided(
        data_2d_padded, shape=shape, strides=strides
    )
