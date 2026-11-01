# 用于执行原始数据清洗分析编码和初步划分
import pandas as pd
import nltk
import os
import json
from pathlib import Path

from preprocessing import (
    category_to_id,
    Training_dataframe,
    split_train_val
)

from data_quality import (
    check_missing_values,
    check_label_conflicts,
    category_distribution,
    handle_missing_values,
    remove_label_conflicts
)
from nltk_resources import ensure_nltk_resources
from data_analysis import (
    split_distribution_comparison,
    text_length_summary_by_category,
    top_words_by_category,
    validate_split_distribution,
)


FINAL_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = (
    FINAL_DIR
    / "data"
    / "raw"
    / "microsoft_text_classification_data.csv"
)
OUTPUT_DIR = FINAL_DIR / "outputs" / "source_data"

def main():
    downloaded_resources, missing_resources = ensure_nltk_resources(
        download_missing=False
    )

    if downloaded_resources:
        print(
            "Downloaded NLTK resources: "
            + ", ".join(downloaded_resources)
        )

    if missing_resources:
        print(
            "Optional NLTK resources are unavailable; "
            "using offline fallbacks for this run: "
            + ", ".join(missing_resources)
        )

    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="utf-8"
    )

    print("\n=== Data Quality Check ===")

    print("\nMissing Values Before Handling:")
    print(check_missing_values(df))

    df = handle_missing_values(df)

    print("\nMissing Values After Handling:")
    print(check_missing_values(df))

    conflicts = check_label_conflicts(df)

    print("\nLabel Conflicts Before Handling:")
    if len(conflicts) == 0:
        print("No label conflicts found.")
    else:
        print(conflicts)

    df = remove_label_conflicts(df)

    print("\nLabel Conflicts After Handling:")
    conflicts_after = check_label_conflicts(df)
    if len(conflicts_after) == 0:
        print("No label conflicts found after handling.")
    else:
        print(conflicts_after)

##################################################
# 清洗数据
##################################################
    df_unique = Training_dataframe(df)

    train_df, val_df = split_train_val(
        df_unique,
        test_size=0.2,
        random_state=666
    )

    print("\n=== NLTK Analysis ===")

    print("\nText length before stopword removal:")
    print(df_unique["text_length"].describe())

    print("\nText length after stopword removal:")
    print(df_unique["text_length_no_stopword"].describe())

    print("\nRemoved stopword count:")
    print(df_unique["removed_stopword_count"].describe())

    from collections import Counter

    all_tokens = [word for tokens in df_unique["tokens_no_stopword"] for word in tokens]
    word_freq = Counter(all_tokens)

    print("\nTop 20 Frequent Words After Stopword Removal:")
    print(word_freq.most_common(20))

    all_pos = [tag for pos_list in df_unique["pos_tags"] for word, tag in pos_list]
    pos_freq = Counter(all_pos)

    print("\nPOS Tag Distribution:")
    print(pos_freq.most_common())

    category_top_words = top_words_by_category(
        df_unique,
        top_n=10,
    )

    print("\nTop 10 Frequent Words By Category:")
    print(category_top_words.to_string(index=False))

    length_summary = text_length_summary_by_category(df_unique)

    print("\nText Length And Stopword Impact By Category:")
    print(length_summary)

    print("\n=== Cleaned Dataset Analysis ===")

    print("\nText Length Statistics:")
    print(df_unique["text_length"].describe())

    print("\nCleaned data shape:")
    print(df_unique.shape)

    print("\nTrain shape:")
    print(train_df.shape)

    print("\nValidation shape:")
    print(val_df.shape)

    print("\nFull Dataset Distribution:")
    print(category_distribution(df_unique))


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_unique.to_csv(OUTPUT_DIR / "cleaned_unique.csv", index=False, encoding="utf-8")
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8")

    with open(OUTPUT_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(category_to_id, f, indent=4)

    print("\n=== Distribution Consistency Check ===")

    distribution_check = split_distribution_comparison(
        full_df=df_unique,
        train_df=train_df,
        val_df=val_df,
    )

    print(distribution_check)

    max_distribution_deviation = validate_split_distribution(
        distribution_check,
        max_deviation_pp=5.0,
    )

    print(
        "Maximum category distribution deviation: "
        f"{max_distribution_deviation:.2f} percentage points"
    )

    category_top_words.to_csv(
        OUTPUT_DIR / "top_words_by_category.csv",
        index=False,
        encoding="utf-8",
    )
    length_summary.to_csv(
        OUTPUT_DIR / "text_length_summary_by_category.csv",
        encoding="utf-8",
    )
    distribution_check.to_csv(
        OUTPUT_DIR / "split_distribution_check.csv",
        encoding="utf-8",
    )

    print("Files saved successfully.")


if __name__ == "__main__":
    main()
