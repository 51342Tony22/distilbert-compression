# 用于划分训练验证测试集并检查数据泄漏
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from runtime import OUTPUT_DIR, SOURCE_DATA_OUTPUT_DIR


def print_distribution(df, dataset_name):
    print(f"\n===== {dataset_name} =====")
    print("Shape:")
    print(df.shape)

    print("\nLabel count:")
    print(df["label_id"].value_counts().sort_index())

    print("\nLabel ratio:")
    print(df["label_id"].value_counts(normalize=True).sort_index())


def validate_split_integrity(original_df, train_df, val_df, test_df):
    """确保样本完整，并检查不同数据集之间不存在文本泄漏。"""

    if len(train_df) + len(val_df) + len(test_df) != len(original_df):
        raise ValueError("Split sizes do not add up to the original dataset.")

    split_texts = {
        "train": set(train_df["text"]),
        "validation": set(val_df["text"]),
        "test": set(test_df["text"]),
    }

    split_names = list(split_texts)

    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            overlap = split_texts[left_name] & split_texts[right_name]

            if overlap:
                raise ValueError(
                    f"Data leakage detected between {left_name} and "
                    f"{right_name}: {len(overlap)} overlapping texts."
                )


def main():
    input_path = SOURCE_DATA_OUTPUT_DIR / "cleaned_unique.csv"
    output_dir = OUTPUT_DIR / "data"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    df = pd.read_csv(
        input_path,
        encoding="utf-8"
    )

    print("Original dataset:")
    print(df.shape)

    print_distribution(
        df,
        "Original Dataset"
    )

    #划分原始数据集1/10为测试集
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=42,
        #保持比例与原有类型近似
        stratify=df["label_id"]
    )

    #需拿出剩下的训练集的1/10作为验证集
    #原训练集有90%，取1/10为1/9
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=1 / 9,
        random_state=42,
        stratify=train_val_df["label_id"]
    )

    validate_split_integrity(
        original_df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    train_path = os.path.join(
        output_dir,
        "train.csv"
    )

    val_path = os.path.join(
        output_dir,
        "val.csv"
    )

    test_path = os.path.join(
        output_dir,
        "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
        encoding="utf-8"
    )

    val_df.to_csv(
        val_path,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        test_path,
        index=False,
        encoding="utf-8"
    )

    print_distribution(
        train_df,
        "Train Set"
    )

    print_distribution(
        val_df,
        "Validation Set"
    )

    print_distribution(
        test_df,
        "Independent Test Set"
    )

    print("\nData split completed successfully.")
    print("Split integrity check passed: no text overlap detected.")
    print("Saved files:")
    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()
