# 用于统一调度数据准备训练量化和测试流程
import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

STAGES = {
    "prepare": [
        PROJECT_ROOT / "prepare_source_data.py",
        PROJECT_ROOT / "prepare_data.py",
    ],
    "train": [
        PROJECT_ROOT / "train_final_model.py",
    ],
    "quantize": [
        PROJECT_ROOT / "evaluate_quantization.py",
    ],
    "test": [
        PROJECT_ROOT / "evaluate_final_model.py",
        PROJECT_ROOT / "evaluate_stability.py",
        PROJECT_ROOT / "evaluate_input_safety.py",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the final text-classification project pipeline."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=[*STAGES, "all"],
        default="all",
        help="Pipeline stage to run. Defaults to the complete pipeline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_stages = list(STAGES) if args.stage == "all" else [args.stage]

    for stage_name in selected_stages:
        for script_path in STAGES[stage_name]:
            print(f"\n===== Running {script_path.name} =====", flush=True)
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
