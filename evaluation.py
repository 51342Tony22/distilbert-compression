# 用于计算验证集整体指标和逐类别指标
import torch

###############################
#增添召回率与精确率
###############################
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)

def evaluate(
    model,
    val_loader,
    device
):
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 返回得分最高的类别标签
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

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    (
        per_class_precision,
        per_class_recall,
        per_class_f1,
        per_class_support,
    ) = precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }

    for label_id in range(3):
        metrics[f"class_{label_id}_precision"] = per_class_precision[label_id]
        metrics[f"class_{label_id}_recall"] = per_class_recall[label_id]
        metrics[f"class_{label_id}_f1"] = per_class_f1[label_id]
        metrics[f"class_{label_id}_support"] = int(
            per_class_support[label_id]
        )

    return metrics
