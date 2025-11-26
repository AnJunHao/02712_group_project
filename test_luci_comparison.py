"""
Compare Luci's implementation with scVelo's original.

This script tests ONLY Luci's parts:
- recover_dynamics()
- latent_time()

Everything else uses scVelo to ensure consistency.
"""
import numpy as np
import scanpy as sc
import scvelo as scv
from cellrank.datasets import bone_marrow

# Import Luci's implementations
from project.tools import recover_dynamics as luci_recover_dynamics
from project.tools import latent_time as luci_latent_time


def main():
    print("="*70)
    print("TESTING LUCI'S IMPLEMENTATION vs scVelo")
    print("="*70)

    # ========================================================================
    # Load and preprocess data (using scVelo/scanpy)
    # ========================================================================
    print("\n[1/5] Loading and preprocessing data with scVelo/scanpy...")
    adata = bone_marrow()
    print(f"   Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    scv.pp.filter_and_normalize(
        adata, min_shared_counts=20, n_top_genes=2000, subset_highly_variable=True
    )
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, random_state=0)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    print(f"   After preprocessing: {adata.n_obs} cells × {adata.n_vars} genes")

    # ========================================================================
    # TEST 1: Luci's recover_dynamics
    # ========================================================================
    print("\n[2/5] Testing Luci's recover_dynamics()...")
    adata_luci = adata.copy()
    luci_recover_dynamics(adata_luci, n_jobs=-1)

    print(f"   ✓ Luci's recover_dynamics completed")
    print(f"   - Genes with successful fits: {adata_luci.var['fit_alpha'].notna().sum()}")
    print(f"   - fit_t layer shape: {adata_luci.layers['fit_t'].shape}")

    # ========================================================================
    # Run scVelo's recover_dynamics for comparison
    # ========================================================================
    print("\n[3/5] Running scVelo's recover_dynamics() for comparison...")
    adata_scv = adata.copy()
    scv.tl.recover_dynamics(adata_scv, n_jobs=-1)

    print(f"   ✓ scVelo's recover_dynamics completed")
    print(f"   - Genes with successful fits: {adata_scv.var['fit_alpha'].notna().sum()}")
    print(f"   - fit_t layer shape: {adata_scv.layers['fit_t'].shape}")

    # ========================================================================
    # Compare recover_dynamics results
    # ========================================================================
    print("\n[4/5] Comparing recover_dynamics outputs...")

    # Compare parameters
    params_to_compare = ['fit_alpha', 'fit_beta', 'fit_gamma', 'fit_likelihood']
    print("\n   Parameter correlations:")
    for param in params_to_compare:
        if param in adata_luci.var.columns and param in adata_scv.var.columns:
            luci_vals = adata_luci.var[param].values
            scv_vals = adata_scv.var[param].values

            # Only compare where both are not NaN
            valid = ~(np.isnan(luci_vals) | np.isnan(scv_vals))
            if np.sum(valid) > 0:
                corr = np.corrcoef(luci_vals[valid], scv_vals[valid])[0, 1]
                print(f"     {param:20s}: {corr:.6f}")
            else:
                print(f"     {param:20s}: No valid values")

    # Compare fit_t layer
    print("\n   fit_t layer comparison:")
    luci_fit_t = adata_luci.layers['fit_t']
    scv_fit_t = adata_scv.layers['fit_t']

    # Compare cell-wise
    valid_cells = ~(np.isnan(luci_fit_t.sum(1)) | np.isnan(scv_fit_t.sum(1)))
    if np.sum(valid_cells) > 0:
        cell_corr = np.corrcoef(
            luci_fit_t[valid_cells].flatten(),
            scv_fit_t[valid_cells].flatten()
        )[0, 1]
        print(f"     Correlation (all values): {cell_corr:.6f}")

    # ========================================================================
    # TEST 2: Luci's latent_time
    # ========================================================================
    print("\n[5/5] Testing Luci's latent_time()...")

    # First compute velocity using scVelo (both need this)
    print("   Computing velocity with scVelo...")
    scv.tl.velocity(adata_luci, mode='dynamical')
    scv.tl.velocity(adata_scv, mode='dynamical')

    # Run Luci's latent_time
    print("   Running Luci's latent_time...")
    luci_latent_time(adata_luci)
    luci_lt = adata_luci.obs['latent_time'].copy()

    # Run scVelo's latent_time
    print("   Running scVelo's latent_time...")
    scv.tl.latent_time(adata_scv)
    scv_lt = adata_scv.obs['latent_time'].copy()

    # ========================================================================
    # Compare latent_time results
    # ========================================================================
    print("\n" + "="*70)
    print("LATENT_TIME COMPARISON RESULTS")
    print("="*70)

    print("\nLuci's latent_time statistics:")
    print(f"  Mean:  {luci_lt.mean():.6f}")
    print(f"  Std:   {luci_lt.std():.6f}")
    print(f"  Min:   {luci_lt.min():.6f}")
    print(f"  Max:   {luci_lt.max():.6f}")
    print(f"  25%:   {np.percentile(luci_lt, 25):.6f}")
    print(f"  50%:   {np.percentile(luci_lt, 50):.6f}")
    print(f"  75%:   {np.percentile(luci_lt, 75):.6f}")

    print("\nscVelo's latent_time statistics:")
    print(f"  Mean:  {scv_lt.mean():.6f}")
    print(f"  Std:   {scv_lt.std():.6f}")
    print(f"  Min:   {scv_lt.min():.6f}")
    print(f"  Max:   {scv_lt.max():.6f}")
    print(f"  25%:   {np.percentile(scv_lt, 25):.6f}")
    print(f"  50%:   {np.percentile(scv_lt, 50):.6f}")
    print(f"  75%:   {np.percentile(scv_lt, 75):.6f}")

    # Compute correlation
    correlation = np.corrcoef(luci_lt, scv_lt)[0, 1]
    print(f"\n{'Pearson correlation:':<25} {correlation:.6f}")

    # Compute differences
    diff = np.abs(luci_lt - scv_lt)
    print(f"{'Mean absolute difference:':<25} {diff.mean():.6f}")
    print(f"{'Max absolute difference:':<25} {diff.max():.6f}")

    # Assessment
    print("\n" + "="*70)
    if correlation > 0.99:
        print("✅ EXCELLENT: Correlation > 0.99 - Implementation matches perfectly!")
    elif correlation > 0.95:
        print("✅ GOOD: Correlation > 0.95 - Implementation is very close!")
    elif correlation > 0.90:
        print("⚠️  WARNING: Correlation > 0.90 - Some discrepancies exist")
    else:
        print("❌ ERROR: Correlation < 0.90 - Significant differences found!")
    print("="*70)


if __name__ == '__main__':
    main()
