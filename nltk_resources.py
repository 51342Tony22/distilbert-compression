# 用于检查和按需下载文本处理所需的 NLTK 资源
import nltk


NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": (
        "taggers/averaged_perceptron_tagger_eng"
    ),
}


def ensure_nltk_resources(download_missing=False):
    """检查可选的 NLTK 数据，并仅在明确要求时下载"""

    downloaded = []
    missing = []

    for package_name, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not download_missing:
                missing.append(package_name)
                continue

            download_succeeded = nltk.download(package_name, quiet=True)

            if download_succeeded:
                downloaded.append(package_name)
            else:
                missing.append(package_name)

    return downloaded, missing
