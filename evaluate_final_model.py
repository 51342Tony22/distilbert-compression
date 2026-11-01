# 用于在独立测试集上比较 FP32 与 INT8 模型
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
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import TextDataset
from model import build_model
from runtime import (
    FINAL_EXPERIMENT_NAME,
    OUTPUT_DIR,
    configure_quantized_backend,
    load_distilbert_tokenizer,
)


def get_model_size_mb(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def measure_inference_time(
    model,
    data_loader,
    device,
    repeat=10
):
    model.eval()

    total_samples = len(data_loader.dataset)

    # 预热运行
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeat):
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

    end_time = time.perf_counter()

    total_time = end_time - start_time

    avg_time_per_run = total_time / repeat
    avg_time_per_sample = total_time / (repeat * total_samples)

    return avg_time_per_run, avg_time_per_sample


def predict(
    model,
    data_loader,
    device
):
    model.eval()

    predictions = []
    true_labels = []

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

    return true_labels, predictions


def evaluate_on_test_set(
    model_name,
    model,
    test_loader,
    device,
    output_dir,
    repeat=10
):
    true_labels, predictions = predict(
        model=model,
        data_loader=test_loader,
        device=device
    )

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

    avg_time_per_run, avg_time_per_sample = measure_inference_time(
        model=model,
        data_loader=test_loader,
        device=device,
        repeat=repeat
    )

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )

    model_path = os.path.join(
        output_dir,
        f"{safe_model_name}_state_dict.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    size_mb = get_model_size_mb(
        model_path
    )

    report = classification_report(
        true_labels,
        predictions,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = os.path.join(
        output_dir,
        f"{safe_model_name}_classification_report.csv"
    )

    report_df.to_csv(
        report_path,
        encoding="utf-8"
    )

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    cm_df = pd.DataFrame(cm)

    cm_path = os.path.join(
        output_dir,
        f"{safe_model_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path,
        index=False,
        encoding="utf-8"
    )

    prediction_df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": predictions
    })

    prediction_path = os.path.join(
        output_dir,
        f"{safe_model_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8"
    )

    result = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "size_mb": size_mb,
        "avg_inference_time_per_run_sec": avg_time_per_run,
        "avg_inference_time_per_sample_sec": avg_time_per_sample,
        "saved_model_path": model_path,
        "classification_report_path": report_path,
        "confusion_matrix_path": cm_path,
        "prediction_path": prediction_path
    }

    return result


def main():
    output_dir = OUTPUT_DIR / "test_results"

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

    # ------------------------------------------------------------
    # 基线模型：
    # 量化前的 FP32 四层结构化剪枝模型。
    # ------------------------------------------------------------
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

    print("\nFP32 4-layer model loaded successfully.")

    # ------------------------------------------------------------
    # 优化模型：
    # 对全部线性层执行 INT8 动态量化的四层模型。
    # ------------------------------------------------------------
    quantized_backend = configure_quantized_backend()
    print(f"Using quantized backend: {quantized_backend}")

    int8_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model),
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    int8_model.eval()

    print("INT8 all-linear quantized model created successfully.")

    results = []

    fp32_result = evaluate_on_test_set(
        model_name="FP32 4-Layer Structured-Pruned DistilBERT",
        model=fp32_model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        repeat=10
    )

    results.append(fp32_result)

    print("\n===== FP32 Test Result =====")
    print(fp32_result)

    int8_result = evaluate_on_test_set(
        model_name="INT8 All-Linear 4-Layer DistilBERT",
        model=int8_model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        repeat=10
    )

    results.append(int8_result)

    print("\n===== INT8 Optimized Test Result =====")
    print(int8_result)

    results_df = pd.DataFrame(results)

    baseline_time = fp32_result["avg_inference_time_per_run_sec"]
    optimized_time = int8_result["avg_inference_time_per_run_sec"]

    baseline_size = fp32_result["size_mb"]
    optimized_size = int8_result["size_mb"]

    baseline_accuracy = fp32_result["accuracy"]
    optimized_accuracy = int8_result["accuracy"]

    speed_improvement_percent = (
        (baseline_time - optimized_time)
        / baseline_time
        * 100
    )

    size_reduction_percent = (
        (baseline_size - optimized_size)
        / baseline_size
        * 100
    )

    accuracy_drop = baseline_accuracy - optimized_accuracy

    target_accuracy = 0.88
    target_speed_improvement = 25.0

    target_accuracy_passed = optimized_accuracy >= target_accuracy
    target_speed_passed = speed_improvement_percent >= target_speed_improvement

    summary = {
        "baseline_model": fp32_result["model"],
        "optimized_model": int8_result["model"],
        "baseline_accuracy": baseline_accuracy,
        "optimized_accuracy": optimized_accuracy,
        "baseline_f1": fp32_result["f1"],
        "optimized_f1": int8_result["f1"],
        "baseline_size_mb": baseline_size,
        "optimized_size_mb": optimized_size,
        "size_reduction_percent": size_reduction_percent,
        "baseline_time_per_run_sec": baseline_time,
        "optimized_time_per_run_sec": optimized_time,
        "speed_improvement_percent": speed_improvement_percent,
        "accuracy_drop": accuracy_drop,
        "target_accuracy": target_accuracy,
        "target_speed_improvement_percent": target_speed_improvement,
        "target_accuracy_passed": target_accuracy_passed,
        "target_speed_passed": target_speed_passed
    }

    summary_df = pd.DataFrame([summary])

    results_path = os.path.join(
        output_dir,
        "final_test_results.csv"
    )

    summary_path = os.path.join(
        output_dir,
        "final_test_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8"
    )

    print("\n===== Final Test Summary =====")
    print(summary_df)

    print("\nSaved final test results to:")
    print(results_path)

    print("Saved final test summary to:")
    print(summary_path)

    if target_accuracy_passed and target_speed_passed:
        print("\nFinal optimized model meets the required targets.")
    else:
        print("\nFinal optimized model does not fully meet the required targets.")
        print("Please review the test summary.")


if __name__ == "__main__":
    main()
