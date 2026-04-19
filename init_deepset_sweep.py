import wandb

MODEL_NAMES = {
    1: "PurePCA",
    2: "FrameAveraging",
    3: "SkewnessCanon",
    4: "RandomFrame",
}


def create_canon_sweep(model_id, project, entity):
    sweep_name = f"ModelNet40_{MODEL_NAMES[model_id]}_RotationStudy"

    sweep_config = {
        "name": sweep_name,
        "program": "train_rot.py",
        "method": "bayes",
        "metric": {"name": "val/best_acc", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 20,
            "eta": 2,
        },
        "parameters": {
            "model": {"value": str(model_id)},

            # --- ADDED EXP_NAME HERE ---
            "exp_name": {"value": MODEL_NAMES[model_id]},

            "epochs": {"value": 100},
            "batch_size": {"values": [32, 64, 128]},
            "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},

            "num_points": {"value": 1024},
            "val_split": {"value": 0.1},
            "seed": {"values": [1, 2, 3, 4, 5]},
            "num_workers": {"value": 4},

            "apply_rotation": {"value": True},  # Reminder: Keep this True for the rotation study!
            "apply_jitter": {"value": False},
            "apply_scale": {"value": False},
            "apply_random_permutation": {"value": False},
            "use_fps": {"value": False},
            "ordering": {"value": "lex"},
        },
        "command": ["${env}", "python", "${program}", "${args}"],
    }

    print(f"Initializing sweep: {sweep_name}")
    return wandb.sweep(sweep=sweep_config, project=project, entity=entity)


def main():
    entity = "team_nadav"
    project = "yon_canon_new"

    sweep_ids = {}
    for model_id in [1, 2, 3, 4]:
        sweep_ids[model_id] = create_canon_sweep(model_id, project, entity)

    print("\n" + "=" * 80)
    print("SWEEPS CREATED:")
    for model_id, sweep_id in sweep_ids.items():
        print(f"Model {model_id} ({MODEL_NAMES[model_id]}): {sweep_id}")
    print("=" * 80 + "\n")

    print("SBATCH commands:")
    print(f"sbatch --job-name=pca_sweep      run_deepset_sweep.sbatch {sweep_ids[1]}")
    print(f"sbatch --job-name=avg_sweep      run_deepset_sweep.sbatch {sweep_ids[2]}")
    print(f"sbatch --job-name=skew_sweep     run_deepset_sweep.sbatch {sweep_ids[3]}")
    print(f"sbatch --job-name=rand_sweep     run_deepset_sweep.sbatch {sweep_ids[4]}")


if __name__ == "__main__":
    main()