# 用于分析高频词文本长度和数据集类别分布
from collections import Counter

import pandas as pd


def top_words_by_category(
    df,
    category_column="category",
    tokens_column="tokens_no_stopword",
    top_n=10,
):
    """返回每个类别中出现频率最高的非停用词"""

    rows = []

    for category, category_df in df.groupby(category_column):
        tokens = [
            token
            for token_list in category_df[tokens_column]
            for token in token_list
        ]

        for rank, (word, count) in enumerate(
            Counter(tokens).most_common(top_n),
            start=1,
        ):
            rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "word": word,
                    "count": count,
                }
            )

    return pd.DataFrame(rows)


def text_length_summary_by_category(df):
    """比较移除停用词前后的文本长度"""

    return (
        df.groupby("category")[
            [
                "text_length",
                "text_length_no_stopword",
                "removed_stopword_count",
            ]
        ]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )


def split_distribution_comparison(full_df, train_df, val_df):
    """比较数据划分前后的类别比例，并计算比例偏差。"""

    def ratio(df):
        return (
            df["category"]
            .value_counts(normalize=True)
            .mul(100)
            .sort_index()
        )

    comparison = pd.DataFrame(
        {
            "full_percent": ratio(full_df),
            "train_percent": ratio(train_df),
            "val_percent": ratio(val_df),
        }
    ).fillna(0)

    comparison["train_deviation_pp"] = (
        comparison["train_percent"] - comparison["full_percent"]
    )
    comparison["val_deviation_pp"] = (
        comparison["val_percent"] - comparison["full_percent"]
    )

    return comparison.round(2)


def validate_split_distribution(comparison, max_deviation_pp=5.0):
    """当划分后的类别分布偏差过大时抛出异常"""

    deviation_columns = [
        "train_deviation_pp",
        "val_deviation_pp",
    ]
    max_observed_deviation = (
        comparison[deviation_columns].abs().max().max()
    )

    if max_observed_deviation > max_deviation_pp:
        raise ValueError(
            "Category distribution drift is too large: "
            f"{max_observed_deviation:.2f} percentage points "
            f"(allowed: {max_deviation_pp:.2f})."
        )

    return max_observed_deviation
