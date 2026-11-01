# 用于重复运行模型并检查指标波动和预测一致性
import os
import time
import copy
import torch
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset import TextDataset
from model import build_model
from runtime import (
    FINAL_EXPERIMENT_NAME,
    OUTPUT_DIR,
    configure_quantized_backend,
    load_distilbert_tokenizer,
)


def predict_and_measure_time(
    model,
    data_loader,
    device
):
    model.eval()

    predictions = []
    true_labels = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    return true_labels, predictions, inference_time


def calculate_metrics(
    true_labels,
    predictions
):
    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    output_dir = OUTPUT_DIR / "stability_results"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    device = torch.device("cpu")

    test_path = OUTPUT_DIR / "data" / "test.csv"

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "checkpoints",
        FINAL_EXPERIMENT_NAME,
        "best_model.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Please run train_final_model.py first."
        )

    test_df = pd.read_csv(
        test_path,
        encoding="utf-8"
    )

    print("Test dataset loaded successfully.")
    print("Test shape:")
    print(test_df.shape)

    tokenizer = load_distilbert_tokenizer()

    test_dataset = TextDataset(
        texts=test_df["clean_text"].tolist(),
        labels=test_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # 加载 FP32 四层模型
    fp32_model = build_model(
        num_labels=3,
        num_hidden_layers=4,
        dropout=0.2,
        attention_dropout=0.2
    )

    fp32_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    fp32_model = fp32_model.to(device)
    fp32_model.eval()

    # 创建 INT8 优化模型
    quantized_backend = configure_quantized_backend()
    print(f"Using quantized backend: {quantized_backend}")

    optimized_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model),
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    optimized_model.eval()

    print("INT8 optimized model loaded successfully.")

    stability_logs = []
    reference_predictions = None

    num_runs = 10

    for run_id in range(1, num_runs + 1):
        print(f"\n===== Stability Test Run {run_id}/{num_runs} =====")

        true_labels, predictions, inference_time = predict_and_measure_time(
            model=optimized_model,
            data_loader=test_loader,
            device=device
        )

        metrics = calculate_metrics(
            true_labels=true_labels,
            predictions=predictions
        )

        avg_time_per_sample = inference_time / len(test_dataset)

        if reference_predictions is None:
            reference_predictions = predictions.copy()

        prediction_changes = sum(
            current != reference
            for current, reference in zip(
                predictions,
                reference_predictions,
            )
        )

        run_result = {
            "run_id": run_id,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "inference_time_sec": inference_time,
            "avg_inference_time_per_sample_sec": avg_time_per_sample,
            "prediction_changes_from_first_run": prediction_changes,
        }

        stability_logs.append(run_result)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Average time per sample: {avg_time_per_sample:.6f} seconds")
        print(f"Prediction changes from first run: {prediction_changes}")

    stability_df = pd.DataFrame(stability_logs)

    stability_path = os.path.join(
        output_dir,
        "stability_test_results.csv"
    )

    stability_df.to_csv(
        stability_path,
        index=False,
        encoding="utf-8"
    )

    summary_rows = []

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "inference_time_sec",
        "avg_inference_time_per_sample_sec"
    ]:
        metric_mean = stability_df[metric].mean()
        metric_std = stability_df[metric].std()
        metric_min = stability_df[metric].min()
        metric_max = stability_df[metric].max()
        metric_range = metric_max - metric_min

        summary_rows.append({
            "metric": metric,
            "mean": metric_mean,
            "std": metric_std,
            "min": metric_min,
            "max": metric_max,
            "range": metric_range
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        output_dir,
        "stability_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8"
    )

    accuracy_range = (
        stability_df["accuracy"].max()
        - stability_df["accuracy"].min()
    )

    recall_range = (
        stability_df["recall"].max()
        - stability_df["recall"].min()
    )

    f1_range = (
        stability_df["f1"].max()
        - stability_df["f1"].min()
    )

    stability_threshold = 0.03

    accuracy_stable = accuracy_range <= stability_threshold
    recall_stable = recall_range <= stability_threshold
    f1_stable = f1_range <= stability_threshold

    final_summary = {
        "accuracy_range": accuracy_range,
        "recall_range": recall_range,
        "f1_range": f1_range,
        "stability_threshold": stability_threshold,
        "accuracy_stable": accuracy_stable,
        "recall_stable": recall_stable,
        "f1_stable": f1_stable,
        "prediction_consistent": (
            stability_df["prediction_changes_from_first_run"].max() == 0
        ),
        "overall_stable": (
            accuracy_stable
            and recall_stable
            and f1_stable
            and stability_df[
                "prediction_changes_from_first_run"
            ].max() == 0
        )
    }

    final_summary_df = pd.DataFrame([final_summary])

    final_summary_path = os.path.join(
        output_dir,
        "stability_final_summary.csv"
    )

    final_summary_df.to_csv(
        final_summary_path,
        index=False,
        encoding="utf-8"
    )

    print("\n===== Stability Summary =====")
    print(summary_df)

    print("\n===== Final Stability Check =====")
    print(final_summary_df)

    print("\nSaved stability test results to:")
    print(stability_path)

    print("Saved stability summary to:")
    print(summary_path)

    print("Saved final stability check to:")
    print(final_summary_path)

    if final_summary["overall_stable"]:
        print("\nModel stability test passed.")
    else:
        print("\nModel stability test failed. Please review the fluctuation range.")


if __name__ == "__main__":
    main()
