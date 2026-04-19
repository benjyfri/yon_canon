import wandb

def create_mlp_sweep(ordering, project, entity):
    sweep_name = f'Global_MLP_{ordering.capitalize()}_noPerm'

    sweep_config = {
        'name': sweep_name,
        'program': 'train.py',
        'method': 'bayes',
        'metric': {'name': 'test/best_acc', 'goal': 'maximize'},
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,
            'eta': 2
        },
        'parameters': {
            'exp_name': {'value': f'mlp_no_aug_{ordering}_noPerm'},
            'model': {'value': 'global_mlp'},
            'ordering': {'value': ordering},
            'epochs': {'value': 100},

            # --- Data & Augmentation Toggles (Explicitly set for clean baseline) ---
            'use_fps': {'value': False}, 
            'apply_jitter': {'value': False},
            'apply_scale': {'value': False},
            'apply_rotation': {'value': False},
            'apply_random_permutation': {'value': False},

            # --- Aggressive Regularization ---
            'batch_size': {'values': [256]},  # Maximize VRAM usage
            'weight_decay': {'values': [1e-3, 1e-2, 5e-2]},  # Heavy L2 penalty
            'dropout': {'values': [0.1, 0.3, 0.5]},
            'label_smoothing': {'values': [0.1, 0.2]},  # Stop overconfident memorization

            # --- Capacity Choking ---
            'num_bands': {'values': [1, 2, 3]},
            'num_points': {'values': [1024]},  # Halve the input dimension!

            # --- Fourier Scale ---
            'fourier_scale': {'values': [0.1, 1.0, 5.0]},

            # --- Learning Rate ---
            'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 5e-3},
        },
        'command': ['${env}', 'python', '${program}', '${args}']
    }

    print(f"Initializing MLP {ordering.upper()} Sweep...")
    return wandb.sweep(sweep_config, project=project, entity=entity)

def main():
    entity = "team_nadav"
    project = "yon_canon_new"

    lex_id = create_mlp_sweep('lex', project, entity)
    hilbert_id = create_mlp_sweep('hilbert', project, entity)
    ply_id = create_mlp_sweep('ply', project, entity)
    pca_id = create_mlp_sweep('pca', project, entity)

    print("\n" + "=" * 70)
    print("✅ MLP SWEEPS CREATED! RUN THE FOLLOWING COMMANDS TO START AGENTS:")
    print("=" * 70)
    # Using the FULL path so the wandb agent can find them!
    print(f"sbatch --job-name=lex_sweep run_sweep.sbatch {lex_id}")
    print(f"sbatch --job-name=hilb_sweep run_sweep.sbatch {hilbert_id}")
    print(f"sbatch --job-name=ply_sweep run_sweep.sbatch {ply_id}")
    print(f"sbatch --job-name=pca_sweep run_sweep.sbatch {pca_id}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()