import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def select_egpcr_coreset(all_embeddings_np, all_recon_errors_np, trial_seed):
    """Select coreset indices using the EGPCR procedure."""
    print("--> Step 2.1: Calculating Sample Quality Utility (Eq. 14-17)...")

    lof_n_neighbors_safe = min(40, len(all_embeddings_np) - 1)
    if lof_n_neighbors_safe <= 0:
        lof_n_neighbors_safe = 1

    lof = LocalOutlierFactor(
        n_neighbors=lof_n_neighbors_safe,
        novelty=False,
        contamination="auto",
    )
    lof.fit(all_embeddings_np)
    l_i = -lof.negative_outlier_factor_

    eps = 1e-8
    z_l_i = (l_i - np.mean(l_i)) / (np.std(l_i) + eps)
    z_e_i = (all_recon_errors_np - np.mean(all_recon_errors_np)) / (
        np.std(all_recon_errors_np) + eps
    )
    a_i = 0.4 * z_l_i + (1.0 - 0.4) * z_e_i

    a_max, a_min = np.max(a_i), np.min(a_i)
    if a_max > a_min:
        u_i = (a_max - a_i) / (a_max - a_min)
    else:
        u_i = np.ones_like(a_i)

    print("--> Step 2.2: BIC-Silhouette GMM Component Selection (Eq. 19-22)...")
    n_samples, _ = all_embeddings_np.shape
    max_k = min(40, int(n_samples / 10), n_samples - 1)
    if max_k < 2:
        max_k = 2

    gmm_models = {}
    bics = {}
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=trial_seed,
        )
        gmm.fit(all_embeddings_np)
        gmm_models[k] = gmm
        bics[k] = gmm.bic(all_embeddings_np)

    delta = 2.0
    b_min = min(bics.values())
    k_delta = {k: bic for k, bic in bics.items() if bic - b_min <= delta}

    best_k = None
    best_sil = -1.0
    for k in k_delta:
        labels = gmm_models[k].predict(all_embeddings_np)
        if 1 < len(np.unique(labels)) < n_samples:
            sil = silhouette_score(all_embeddings_np, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k

    if best_k is None:
        best_k = min(bics, key=bics.get)

    print(f"    Selected optimal GMM components K* = {best_k}")
    final_gmm = gmm_models[best_k]

    print("--> Step 2.3: Computing Reference Weights (Eq. 23-25)...")
    gamma = final_gmm.predict_proba(all_embeddings_np)
    n_k = np.sum(gamma, axis=0)
    n_k[n_k == 0] = eps
    b_j = (1.0 / best_k) * np.sum(gamma / n_k, axis=1)

    beta_numerator = u_i * b_j
    beta_j = beta_numerator / (np.sum(beta_numerator) + eps)

    print("--> Step 2.4: Submodular Greedy Coreset Selection (Algorithm 1, Eq. 26-28)...")
    m_budget = max(1, int(0.90 * n_samples))

    scaler_latent = StandardScaler().fit(all_embeddings_np)
    h_bar = scaler_latent.transform(all_embeddings_np)

    dists = cdist(h_bar, h_bar, metric="sqeuclidean")
    nonzero_dists = dists[dists > 1e-8]
    sigma_h_sq = np.median(nonzero_dists) if len(nonzero_dists) > 0 else 1.0
    kappa = np.exp(-dists / (2.0 * sigma_h_sq))

    selected_indices = []
    available = np.ones(n_samples, dtype=bool)
    c_j = np.zeros(n_samples)

    quality_term = ((1.0 - 0.31981631126330323) / m_budget) * u_i

    for _ in range(m_budget):
        uncovered_gain = np.maximum(0, kappa - c_j[:, None])
        coverage_term = 0.31981631126330323 * np.dot(beta_j, uncovered_gain)

        delta_i = quality_term + coverage_term
        delta_i[~available] = -np.inf
        best_i = np.argmax(delta_i)

        selected_indices.append(best_i)
        available[best_i] = False
        c_j = np.maximum(c_j, kappa[:, best_i])

    return np.array(selected_indices)
