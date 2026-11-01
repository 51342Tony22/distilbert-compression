# 用于配置并训练最终结构化剪枝文本分类模型
import argparse
import os
import pandas as pd
import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from dataset import TextDataset
from model import build_model, print_model_info
from training import train_one_epoch
from evaluation import evaluate
from augmentation import augment_training_data_with_synonyms
from runtime import (
    FINAL_EXPERIMENT_NAME,
    OUTPUT_DIR,
    get_training_device,
    load_distilbert_tokenizer,
)


def get_available_path(base_path):

    if not os.path.exists(base_path):
        return base_path

    name, ext = os.path.splitext(base_path)

    counter = 1

    while True:
        new_path = f"{name}({counter}){ext}"

        if not os.path.exists(new_path):
            return new_path

        counter += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the final cross-platform DistilBERT classifier."
    )
    parser.add_argument(
        "--experiment-name",
        default=FINAL_EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--scheduler",
        choices=["warmup_cosine", "cosine", "constant"],
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    #最后参数
    experiment_name = args.experiment_name

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = 1e-5
    weight_decay = args.weight_decay
    max_length = 128

    num_hidden_layers = 4
    dropout = 0.2
    attention_dropout = 0.2

    augment_ratio = 0.3
    warmup_ratio = args.warmup_ratio


    train_path = OUTPUT_DIR / "data" / "train.csv"

    val_path = OUTPUT_DIR / "data" / "val.csv"

    checkpoint_dir = os.path.join(
        OUTPUT_DIR,
        "checkpoints",
        experiment_name
    )

    log_dir = os.path.join(
        OUTPUT_DIR,
        "logs"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    os.makedirs(
        log_dir,
        exist_ok=True
    )

    log_path = get_available_path(
        os.path.join(
            log_dir,
            f"{experiment_name}_training_log.csv"
        )
    )

    #读取数据
    train_df = pd.read_csv(
        train_path,
        encoding="utf-8"
    )

    val_df = pd.read_csv(
        val_path,
        encoding="utf-8"
    )

    print("Original train shape:")
    print(train_df.shape)

    print("Validation shape:")
    print(val_df.shape)


    train_df, augmented_count = augment_training_data_with_synonyms(
        train_df=train_df,
        text_column="clean_text",
        augment_ratio=augment_ratio,
        random_state=42
    )

    print("Synonym replacement augmentation enabled.")
    print(f"Augmented samples added: {augmented_count}")

    print("Train shape after augmentation:")
    print(train_df.shape)

    tokenizer = load_distilbert_tokenizer()


    train_dataset = TextDataset(
        texts=train_df["clean_text"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = TextDataset(
        texts=val_df["clean_text"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    device = get_training_device()

    print("Using device:")
    print(device)


    model = build_model(
        num_labels=3,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        attention_dropout=attention_dropout
    )

    print_model_info(model)

    model = model.to(device)


    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


    total_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(total_training_steps * warmup_ratio)

    if args.scheduler == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps
        )
    elif args.scheduler == "cosine":
        num_warmup_steps = 0
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_training_steps
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
        )

    print("Scheduler type:")
    print(args.scheduler)

    print("Total training steps:")
    print(total_training_steps)

    print("Warmup steps:")
    print(num_warmup_steps)

    # 早停机制
    best_val_accuracy = float("-inf")
    patience = 5
    patience_counter = 0
    min_epochs = 8

    training_logs = []

    print("Experiment name:")
    print(experiment_name)

    print("Training log will be saved to:")
    print(log_path)

    print("Checkpoints will be saved to:")
    print(checkpoint_dir)

    #训练循环
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )

        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            device=device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation Precision: {val_metrics['precision']:.4f}")
        print(f"Validation Recall: {val_metrics['recall']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Learning Rate: {current_lr:.8f}")

        training_logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "class_0_precision": val_metrics["class_0_precision"],
            "class_0_recall": val_metrics["class_0_recall"],
            "class_0_f1": val_metrics["class_0_f1"],
            "class_1_precision": val_metrics["class_1_precision"],
            "class_1_recall": val_metrics["class_1_recall"],
            "class_1_f1": val_metrics["class_1_f1"],
            "class_2_precision": val_metrics["class_2_precision"],
            "class_2_recall": val_metrics["class_2_recall"],
            "class_2_f1": val_metrics["class_2_f1"],
            "learning_rate": current_lr
        })

        log_df = pd.DataFrame(training_logs)

        log_df.to_csv(
            log_path,
            index=False,
            encoding="utf-8"
        )

        current_val_accuracy = val_metrics["accuracy"]
        stop_training = False

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            patience_counter = 0

            best_checkpoint_path = os.path.join(
                checkpoint_dir,
                "best_model.pt"
            )

            torch.save(
                model.state_dict(),
                best_checkpoint_path
            )

            print(
                f"Best model updated. "
                f"Best Validation Accuracy: {best_val_accuracy:.4f}"
            )

        else:
            if epoch + 1 >= min_epochs:
                patience_counter += 1

                print(
                    f"No improvement. "
                    f"Patience counter: {patience_counter}/{patience}"
                )

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    stop_training = True

            else:
                print(
                    f"No improvement, but early stopping is disabled "
                    f"before epoch {min_epochs}."
                )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            "latest_model.pt"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(f"Checkpoint saved to: {checkpoint_path}")

        if stop_training:
            break

    print("\nTraining finished successfully.")
    print("Best validation accuracy:")
    print(best_val_accuracy)


if __name__ == "__main__":
    main()
