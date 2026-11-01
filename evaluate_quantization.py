# 用于比较不同动态量化方案的精度体积和速度
import os
import time
import torch
import copy
import pandas as pd

from torch.utils.data import DataLoader
from dataset import TextDataset
from model import build_model
from evaluation import evaluate
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

def count_trainable_parameters(model):
    total_params = 0

    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()

    return total_params


def get_transformer_layer_count(model):
    return len(model.distilbert.transformer.layer)


#计算跑完验证集需要多久
def measure_inference_time(
    model,
    data_loader,
    device,
    repeat=5
):
    model.eval()

    total_samples = len(data_loader.dataset)

    #空跑一次
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

    average_time_per_run = total_time / repeat
    average_time_per_sample = total_time / (repeat * total_samples)

    return average_time_per_run, average_time_per_sample
def quantize_all_linear_layers(model):
    configure_quantized_backend()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    return quantized_model

def quantize_selected_linear_layers(
    model,
    target_keywords
):
    configure_quantized_backend()

    qconfig_spec = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for keyword in target_keywords:
                if keyword in name:
                    qconfig_spec[name] = torch.quantization.default_dynamic_qconfig

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8
    )

    return quantized_model

def evaluate_model_variant(
    model_name,
    model,
    val_loader,
    device,
    output_dir,
    repeat=5
):
    model.eval()

    metrics = evaluate(
        model=model,
        val_loader=val_loader,
        device=device
    )

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

    model_path = os.path.join(
        output_dir,
        f"{safe_model_name}_state_dict.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    model_size = get_model_size_mb(model_path)

    time_per_run, time_per_sample = measure_inference_time(
        model=model,
        data_loader=val_loader,
        device=device,
        repeat=repeat
    )

    result = {
        "model": model_name,
        "size_mb": model_size,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "avg_inference_time_per_run_sec": time_per_run,
        "avg_inference_time_per_sample_sec": time_per_sample,
        "saved_model_path": model_path
    }

    return result



def main():
    output_dir = OUTPUT_DIR / "quantization"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")
    # 使用完成数据增强训练后的最佳检查点进行量化
    experiment_name = FINAL_EXPERIMENT_NAME
    num_hidden_layers = 4

    best_model_path = os.path.join(
        OUTPUT_DIR,
        "checkpoints",
        experiment_name,
        "best_model.pt"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found: {best_model_path}. "
            f"Please run train_final_model.py for {experiment_name} first."
        )

    #加载验证集
    val_path = OUTPUT_DIR / "data" / "val.csv"

    val_df = pd.read_csv(
        val_path,
        encoding="utf-8"
    )

    tokenizer = load_distilbert_tokenizer()

    val_dataset = TextDataset(
        texts=val_df["clean_text"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    print("Validation dataset loaded successfully.")
    print("Validation size:")
    print(len(val_dataset))

    #加载原模型 FP32
    fp32_model = build_model(
        num_labels=3,
        num_hidden_layers=num_hidden_layers,
        dropout=0.2,
        attention_dropout=0.2
    )

    fp32_model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    fp32_model = fp32_model.to(device)
    fp32_model.eval()

    print("\n===== Structured-Pruned FP32 Model Information =====")
    print(f"Experiment name: {experiment_name}")
    print(f"Transformer layers: {get_transformer_layer_count(fp32_model)}")
    print(f"Trainable parameters: {count_trainable_parameters(fp32_model):,}")
    print(f"Loaded checkpoint: {best_model_path}")

    comparison_results = []

    fp32_result = evaluate_model_variant(
        model_name="Structured-Pruned FP32 4-Layer DistilBERT",
        model=fp32_model,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        repeat=5
    )

    comparison_results.append(fp32_result)

    print("\n===== FP32 Structured-Pruned Model Evaluation =====")
    print(fp32_result)

    all_linear_model = quantize_all_linear_layers(
        copy.deepcopy(fp32_model)
    )

    all_linear_result = evaluate_model_variant(
        model_name="INT8 All-Linear Quantized 4-Layer DistilBERT",
        model=all_linear_model,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        repeat=5
    )

    comparison_results.append(all_linear_result)

    print("\n===== INT8 All-Linear Quantized Evaluation =====")
    print(all_linear_result)

    #
    attention_model = quantize_selected_linear_layers(
        model=copy.deepcopy(fp32_model),
        target_keywords=[
            "attention.q_lin",
            "attention.k_lin",
            "attention.v_lin",
            "attention.out_lin"
        ]
    )

    attention_result = evaluate_model_variant(
        model_name="INT8 Attention-Only Quantized 4-Layer DistilBERT",
        model=attention_model,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        repeat=5
    )

    comparison_results.append(attention_result)

    print("\n===== INT8 Attention-Only Quantized Evaluation =====")
    print(attention_result)

    # 4. 仅量化前馈神经网络层
    ffn_model = quantize_selected_linear_layers(
        model=copy.deepcopy(fp32_model),
        target_keywords=[
            "ffn.lin1",
            "ffn.lin2"
        ]
    )

    ffn_result = evaluate_model_variant(
        model_name="INT8 FFN-Only Quantized 4-Layer DistilBERT",
        model=ffn_model,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        repeat=5
    )

    comparison_results.append(ffn_result)

    print("\n===== INT8 FFN-Only Quantized Evaluation =====")
    print(ffn_result)

    #
    classifier_model = quantize_selected_linear_layers(
        model=copy.deepcopy(fp32_model),
        target_keywords=[
            "pre_classifier",
            "classifier"
        ]
    )

    classifier_result = evaluate_model_variant(
        model_name="INT8 Classifier-Only Quantized 4-Layer DistilBERT",
        model=classifier_model,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        repeat=5
    )

    comparison_results.append(classifier_result)

    print("\n===== INT8 Classifier-Only Quantized Evaluation =====")
    print(classifier_result)

    comparison_df = pd.DataFrame(comparison_results)

    baseline_size = fp32_result["size_mb"]
    baseline_time = fp32_result["avg_inference_time_per_run_sec"]
    baseline_accuracy = fp32_result["accuracy"]
    baseline_f1 = fp32_result["f1"]

    comparison_df["size_reduction_percent"] = (
            (baseline_size - comparison_df["size_mb"])
            / baseline_size
            * 100
    )

    comparison_df["speed_improvement_percent"] = (
            (baseline_time - comparison_df["avg_inference_time_per_run_sec"])
            / baseline_time
            * 100
    )

    comparison_df["accuracy_drop"] = (
            baseline_accuracy - comparison_df["accuracy"]
    )

    comparison_df["f1_drop"] = (
            baseline_f1 - comparison_df["f1"]
    )

    comparison_path = os.path.join(
        output_dir,
        "quantization_sensitivity_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8"
    )

    print("\n===== Quantization Sensitivity Comparison Table =====")
    print(comparison_df)

    print("\nComparison saved to:")
    print(comparison_path)

    # 选择候选模型：
    # 准确率下降应较小，同时推理速度必须有所提升
    candidate_df = comparison_df[
        (comparison_df["model"] != fp32_result["model"])
        & (comparison_df["accuracy_drop"] <= 0.03)
        & (comparison_df["speed_improvement_percent"] > 0)
    ].copy()

    if len(candidate_df) > 0:
        candidate_df = candidate_df.sort_values(
            by=[
                "speed_improvement_percent",
                "size_reduction_percent"
            ],
            ascending=False
        )

        best_candidate = candidate_df.iloc[0]

        print("\n===== Recommended Final Optimized Model =====")
        print(f"Model: {best_candidate['model']}")
        print(f"Accuracy: {best_candidate['accuracy']:.4f}")
        print(f"F1: {best_candidate['f1']:.4f}")
        print(f"Size reduction: {best_candidate['size_reduction_percent']:.2f}%")
        print(f"Speed improvement: {best_candidate['speed_improvement_percent']:.2f}%")
        print(f"Accuracy drop: {best_candidate['accuracy_drop']:.4f}")
        print(f"Saved model path: {best_candidate['saved_model_path']}")
    else:
        print(
            "\nNo quantized model satisfies accuracy_drop <= 0.03 "
            "and positive speed improvement on this platform."
        )
        print("Please review the comparison table manually.")


if __name__ == "__main__":
    main()
