# 用于清除文本中的 HTML 链接符号和缩写噪声
import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r"https?://\S+|www\S+", " ", text)
    text = re.sub(r"\bcan['’]t\b", "cannot", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwon['’]t\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"n['’]t\b", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"['’]re\b", " are", text, flags=re.IGNORECASE)
    text = re.sub(r"['’]ve\b", " have", text, flags=re.IGNORECASE)
    text = re.sub(r"['’]ll\b", " will", text, flags=re.IGNORECASE)
    text = re.sub(r"['’]m\b", " am", text, flags=re.IGNORECASE)
    text = re.sub(r"['’]s\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s]"," ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


#if __name__ == "__main__":
    df = pd.read_csv("microsoft_text_classification_data.csv",encoding="utf-8")
    df_unique = df.drop_duplicates()
    df_unique["clean_text"] = df_unique["text"].apply(clean_text)
    print(df_unique[["text", "clean_text","category"]].head().to_string())

    def count_words(text):
        words = text.split()
        return len(words)

    df_unique["text_length"] = df_unique["clean_text"].apply(count_words)

    print(df_unique[["clean_text", "text_length", "category"]].head().to_string())
    print(df_unique["text_length"].describe())
    print(df_unique.sort_values("text_length", ascending=False)[["clean_text", "text_length", "category"]].head(10).to_string())
