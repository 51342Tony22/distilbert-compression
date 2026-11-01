# 用于批量运行超参数实验并汇总最佳结果
import subprocess
import sys

import pandas as pd

from runtime import FINAL_DIR, OUTPUT_DIR


EXPERIMENTS = [
    {
        "experiment_name": "compare_cosine_wd0001_bs32",
        "scheduler": "cosine",
        "warmup_ratio": 0.0,
        "weight_decay": 0.001,
        "batch_size": 32,
    },
    {
        "experiment_name": "compare_warmup_cosine_wd0001_bs32",
        "scheduler": "warmup_cosine",
        "warmup_ratio": 0.10,
        "weight_decay": 0.001,
        "batch_size": 32,
    },
    {
        "experiment_name": "compare_warmup_cosine_wd001_bs32",
        "scheduler": "warmup_cosine",
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "batch_size": 32,
    },
    {
        "experiment_name": "compare_constant_wd0001_bs16",
        "scheduler": "constant",
        "warmup_ratio": 0.10,
        "weight_decay": 0.001,
        "batch_size": 16,
    },
]


def latest_log_path(experiment_name):
    candidates = list(
        (OUTPUT_DIR / "logs").glob(
            f"{experiment_name}_training_log*.csv"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"No training log found for {experiment_name}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_experiment(config):
    log_path = latest_log_path(config["experiment_name"])
    log_df = pd.read_csv(log_path)
    best_row = log_df.loc[log_df["val_accuracy"].idxmax()]

    return {
        **config,
        "best_epoch": int(best_row["epoch"]),
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_macro_f1": best_row["val_macro_f1"],
        "final_learning_rate": log_df.iloc[-1]["learning_rate"],
        "epochs_completed": len(log_df),
        "log_path": str(log_path),
    }


def main():
    train_script = FINAL_DIR / "train_final_model.py"
    summaries = []

    for config in EXPERIMENTS:
        command = [
            sys.executable,
            str(train_script),
            "--experiment-name",
            config["experiment_name"],
            "--scheduler",
            config["scheduler"],
            "--warmup-ratio",
            str(config["warmup_ratio"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--batch-size",
            str(config["batch_size"]),
        ]

        print(f"Running experiment: {config['experiment_name']}")
        subprocess.run(command, check=True)
        summaries.append(summarize_experiment(config))

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["best_val_accuracy", "best_val_macro_f1"],
        ascending=False,
    )

    output_path = (
        OUTPUT_DIR
        / "logs"
        / "hyperparameter_comparison.csv"
    )
    summary_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n===== Hyperparameter Comparison =====")
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()
