import os
import random

import joblib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import (
    BAYESIAN_EPSILON,
    COLUMNS_TO_DELETE_ORIG,
    DATA_DIR,
    DEVICE,
    FAULT_IDS,
    MODEL_DIR,
    MODEL_FILENAME,
    N_NORMAL_SAMPLES,
    NUM_FEATURES,
    SCALER_FILENAME,
    TEST_DATA_FILENAME,
    TRAIN_DATA_FILENAME,
    UNCERTAINTY_LEVELS,
    UNCERTAIN_VARIABLES_INDICES,
)
from .coreset import select_egpcr_coreset
from .data_utils import (
    inject_multiplicative_noise,
    load_and_preprocess_train_data,
    prepare_global_windows_padded,
)
from .metrics import calculate_far_fdr, spekde
from .model import GlobalTransformer


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(trial_seed):
    set_seed(trial_seed)

    print("\n--- Data Preparation ---")
    clean_train_raw, scaler, n_features = load_and_preprocess_train_data(
        DATA_DIR, TRAIN_DATA_FILENAME, COLUMNS_TO_DELETE_ORIG
    )
    if n_features != NUM_FEATURES:
        print(
            f"Warning: Loaded data has {n_features} features, "
            f"but expected {NUM_FEATURES}."
        )

    print(f"Injecting noise into variables: {UNCERTAIN_VARIABLES_INDICES}")
    noisy_train_raw = inject_multiplicative_noise(
        clean_train_raw,
        UNCERTAIN_VARIABLES_INDICES,
        UNCERTAINTY_LEVELS,
        trial_seed,
    )

    clean_train_scaled = scaler.transform(clean_train_raw)
    noisy_train_scaled = scaler.transform(noisy_train_raw)

    clean_train_windows_np = prepare_global_windows_padded(
        clean_train_scaled, window_size=16
    )
    noisy_train_windows_np = prepare_global_windows_padded(
        noisy_train_scaled, window_size=16
    )

    model = GlobalTransformer(
        window_size=16,
        num_global_features=n_features,
        d_model=64,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=96,
        dropout=0.1033567581461701,
    ).to(DEVICE)

    print("\n--- Stage 1: Initial Denoising Training ---")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(
        torch.from_numpy(noisy_train_windows_np).float(),
        torch.from_numpy(clean_train_windows_np).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for _ in tqdm(range(60), desc="Denoising Training", leave=False):
        model.train()
        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(DEVICE)
            clean_batch = clean_batch.to(DEVICE)

            optimizer.zero_grad()

            # 保持原脚本行为：输入 noisy_batch，训练目标也是 noisy_batch。
            training_loss = model(noisy_batch, target_window=noisy_batch)
            training_loss.backward()
            optimizer.step()

    print("\n--- Stage 2: EGPCR Coreset Selection ---")
    model.eval()
    all_embeddings = []
    all_recon_errors = []

    unshuffled_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for noisy_batch, clean_batch in tqdm(
            unshuffled_loader,
            desc="Extracting Embeddings & Errors",
            leave=False,
        ):
            noisy_batch = noisy_batch.to(DEVICE)
            clean_batch = clean_batch.to(DEVICE)

            spe_losses, embeddings = model(
                noisy_batch,
                target_window=noisy_batch,
                return_embedding=True,
            )
            all_embeddings.append(embeddings.cpu().numpy())
            all_recon_errors.append(spe_losses.cpu().numpy())

    all_embeddings_np = np.concatenate(all_embeddings)
    all_recon_errors_np = np.concatenate(all_recon_errors)

    core_set_indices = select_egpcr_coreset(
        all_embeddings_np,
        all_recon_errors_np,
        trial_seed,
    )

    coreset_noisy_windows = noisy_train_windows_np[core_set_indices]
    coreset_clean_windows = clean_train_windows_np[core_set_indices]

    print(f"Original training windows: {len(noisy_train_windows_np)}")
    print(
        f"Selected {len(coreset_noisy_windows)} diverse windows "
        f"for coreset (~{90.0:.1f}%)"
    )

    print("\n--- Stage 3: Refinement Training ---")
    refine_optimizer = optim.Adam(model.parameters(), lr=0.001 / 10)

    coreset_dataset = TensorDataset(
        torch.from_numpy(coreset_noisy_windows).float(),
        torch.from_numpy(coreset_clean_windows).float(),
    )
    effective_batch_size = min(32, len(coreset_dataset))
    coreset_loader = DataLoader(
        coreset_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
    )

    for _ in tqdm(range(50), desc="Refining", leave=False):
        model.train()
        for noisy_batch, clean_batch in coreset_loader:
            noisy_batch = noisy_batch.to(DEVICE)
            clean_batch = clean_batch.to(DEVICE)

            refine_optimizer.zero_grad()

            # 保持原脚本行为：输入 noisy_batch，训练目标也是 noisy_batch。
            refine_loss = model(noisy_batch, target_window=noisy_batch)
            refine_loss.backward()
            refine_optimizer.step()

    print("\n--- Stage 4 & 5: Threshold Calculation and Final Evaluation ---")
    model.eval()

    with torch.no_grad():
        coreset_tensor = torch.from_numpy(coreset_noisy_windows).float().to(DEVICE)
        coreset_spe = model(
            coreset_tensor,
            target_window=coreset_tensor,
        ).cpu().numpy()

        threshold_spe = max(
            BAYESIAN_EPSILON,
            spekde(
                coreset_spe,
                bw=0.17,
                q=0.98,
                seed=trial_seed,
            ),
        )

    all_fault_results = {}
    print("\n--- Evaluating Faults ---")

    for fault_id in tqdm(FAULT_IDS, desc="Evaluating Faults", leave=False):
        test_file = os.path.join(DATA_DIR, TEST_DATA_FILENAME)
        if not os.path.exists(test_file):
            continue

        raw_test_data = np.loadtxt(test_file)
        if COLUMNS_TO_DELETE_ORIG:
            valid_cols = [
                c for c in COLUMNS_TO_DELETE_ORIG if c < raw_test_data.shape[1]
            ]
            raw_test_data = np.delete(raw_test_data, valid_cols, axis=1)

        noisy_raw_test_data = inject_multiplicative_noise(
            raw_test_data,
            UNCERTAIN_VARIABLES_INDICES,
            UNCERTAINTY_LEVELS,
            trial_seed + fault_id,
        )

        scaled_clean_test_data = scaler.transform(raw_test_data)
        scaled_noisy_test_data = scaler.transform(noisy_raw_test_data)

        clean_test_windows = prepare_global_windows_padded(
            scaled_clean_test_data, window_size=16
        )
        noisy_test_windows = prepare_global_windows_padded(
            scaled_noisy_test_data, window_size=16
        )

        if noisy_test_windows.shape[0] == 0:
            continue

        num_normal_samples = N_NORMAL_SAMPLES
        num_fault_samples = noisy_test_windows.shape[0] - num_normal_samples

        with torch.no_grad():
            noisy_test_tensor = torch.from_numpy(noisy_test_windows).float().to(DEVICE)
            spe_test_vals = model(
                noisy_test_tensor,
                target_window=noisy_test_tensor,
            ).cpu().numpy()

        alarms = (spe_test_vals > threshold_spe).astype(int)
        far, fdr = calculate_far_fdr(
            alarms,
            num_normal_samples,
            num_fault_samples,
        )
        all_fault_results[fault_id] = {"far": far, "fdr": fdr}

    avg_far = (
        np.nanmean([res["far"] for res in all_fault_results.values() if res])
        if all_fault_results
        else 0.0
    )
    avg_fdr = (
        np.nanmean([res["fdr"] for res in all_fault_results.values() if res])
        if all_fault_results
        else 0.0
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_FILENAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILENAME))

    return avg_far, avg_fdr, all_fault_results
