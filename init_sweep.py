import wandb


def create_sweep(ordering, project, entity):
    # Give indicative names based on the ordering
    sweep_name = f'RoPE_PT_{ordering.capitalize()}_Optimization'

    sweep_config = {
        'name': sweep_name,
        'program': 'train.py',
        'method': 'bayes',
        'metric': {'name': 'test/best_acc', 'goal': 'maximize'},
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 30,
            'eta': 2
        },
        'parameters': {
            # Give runs indicative prefix names (e.g., lex_run_1, hilbert_run_2)
            'exp_name': {'value': f'pt_rope_{ordering}_noPerm'},
            'model': {'value': 'point_transformer'},

            # --- LOCKED TO THE SPECIFIC ORDERING ---
            'ordering': {'value': ordering},

            'epochs': {'value': 150},
            'batch_size': {'value': 32},

            # --- Optimizer & LR ---
            'optimizer': {'value': 'adamw'},
            'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 5e-3},
            'weight_decay': {'values': [1e-4, 1e-3, 1e-2]},

            # --- Architecture (Locked) ---
            'trans_dim': {'value': 216},
            'trans_depth': {'value': 4},
            'trans_heads': {'value': 6},

            # --- Regularization ---
            'dropout': {'values': [0.1, 0.2, 0.3]},
            'drop_path_rate': {'values': [0.05, 0.1, 0.2]},
            'label_smoothing': {'values': [0.0, 0.1, 0.2]}
        },
        'command': ['${env}', 'python', '${program}', '${args}']
    }

    print(f"Initializing {ordering.upper()} Sweep...")
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    return sweep_id


def main():
    entity = "team_nadav"
    project = "yon_canon"  # Updated project name!

    # Create both sweeps
    lex_id = create_sweep('lex', project, entity)
    hilbert_id = create_sweep('hilbert', project, entity)
    ply_id = create_sweep('ply', project, entity)

    print("\n" + "=" * 65)
    print("✅ SWEEPS CREATED SUCCESSFULLY!")
    print(f"👉 LEXICOGRAPHICAL Sweep ID : {lex_id}")
    print(f"👉 HILBERT CURVE Sweep ID   : {hilbert_id}")
    print(f"👉 Regular Sweep ID   : {ply_id}")
    print("=" * 65 + "\n")

    print("To launch your Slurm workers, use:")
    print(f"sbatch --job-name=lex_sweep run_sweep.sbatch {lex_id}")
    print(f"sbatch --job-name=hilb_sweep run_sweep.sbatch {hilbert_id}\n")
    print(f"sbatch --job-name=clean_sweep run_sweep.sbatch {ply_id}\n")


if __name__ == "__main__":
    main()