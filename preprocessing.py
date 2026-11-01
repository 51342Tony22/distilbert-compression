# 用于完成文本分词词性标注停用词处理和标签编码
import pandas as pd
import nltk
from functools import lru_cache

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from text_cleaning import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

category_to_id = {
    "complaint_suggestion": 0,
    "feature_feedback": 1,
    "problem_consultation": 2
}

def tokenize_text(text):
    return nltk.word_tokenize(text, preserve_line=True)

def pos_tag(text):
    try:
        return nltk.pos_tag(text)
    except LookupError:
        return nltk.DefaultTagger("UNK").tag(text)

@lru_cache(maxsize=1)
def get_english_stopwords():
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set(ENGLISH_STOP_WORDS)

    stop_words.update({"s", "t", "d", "ll", "m", "re", "ve"})
    return stop_words


def remove_stopwords(text):
    stop_words = get_english_stopwords()
    return [word for word in text if word not in stop_words]


#df = pd.read_csv("microsoft_text_classification_data.csv",encoding="utf-8")

def Training_dataframe(df):
    df_unique = df.drop_duplicates().copy()

    df_unique["clean_text"] = df_unique["text"].apply(clean_text)

    df_unique["tokens"] = df_unique["clean_text"].apply(tokenize_text)
    df_unique["pos_tags"] = df_unique["tokens"].apply(pos_tag)
    df_unique["tokens_no_stopword"] = df_unique["tokens"].apply(remove_stopwords)

    df_unique["text_length"] = df_unique["tokens"].apply(len)
    df_unique["text_length_no_stopword"] = df_unique["tokens_no_stopword"].apply(len)
    df_unique["removed_stopword_count"] = (
            df_unique["text_length"] - df_unique["text_length_no_stopword"]
    )

    df_unique["label_id"] = df_unique["category"].map(category_to_id)

    if df_unique["label_id"].isnull().any():
        unknown_labels = df_unique[df_unique["label_id"].isnull()]["category"].unique()
        raise ValueError(f"Unknown category found: {unknown_labels}")

    return df_unique


def split_train_val(df_unique, test_size, random_state):
    train_df, val_df = train_test_split(
        df_unique,
        test_size=test_size,
        random_state=random_state,
        stratify=df_unique["category"]
    )
    return train_df, val_df

#print("训练集形状：")
#print(train_df.shape)

#print("验证集形状：")
#print(val_df.shape)

#print("训练集类别分布：")
#print(train_df["category"].value_counts())

#print("验证集类别分布：")
#print(val_df["category"].value_counts())
