# 用于通过同义词替换扩充训练数据
import random
import pandas as pd

#同义词替换
SYNONYM_MAP = {
    "slow": ["sluggish", "delayed"],
    "bad": ["poor"],
    "terrible": ["awful"],
    "crash": ["fail"],
    "crashes": ["fails"],
    "crashing": ["failing"],
    "error": ["issue"],
    "problem": ["issue"],
    "issue": ["problem"],
    "feature": ["function"],
    "function": ["feature"],
    "useful": ["helpful"],
    "improve": ["enhance"],
    "improved": ["enhanced"],
    "support": ["assist"],
    "suggestion": ["recommendation"],
    "feedback": ["comment"]
}


PROTECTED_WORDS = {
    "not", "no", "never", "cannot", "cant",
    "how", "why", "where", "when", "what",
    "can", "could", "should", "would"
}


def synonym_replace_text(
    text,
    max_replacements=1
):
    words = str(text).split()

    candidate_indices = []

    for index, word in enumerate(words):
        if word in SYNONYM_MAP and word not in PROTECTED_WORDS:
            candidate_indices.append(index)

    if len(candidate_indices) == 0:
        return text

    random.shuffle(candidate_indices)

    replacement_count = 0

    for index in candidate_indices:
        original_word = words[index]
        synonyms = SYNONYM_MAP[original_word]

        new_word = random.choice(synonyms)

        words[index] = new_word
        replacement_count += 1

        if replacement_count >= max_replacements:
            break

    return " ".join(words)


def augment_training_data_with_synonyms(
    train_df,
    text_column="clean_text",
    augment_ratio=0.3,
    random_state=42
):
    random.seed(random_state)

    sampled_df = train_df.sample(
        frac=augment_ratio,
        random_state=random_state
    )

    augmented_rows = []

    for _, row in sampled_df.iterrows():
        original_text = row[text_column]

        augmented_text = synonym_replace_text(
            original_text,
            max_replacements=1
        )

        if augmented_text != original_text:
            new_row = row.copy()
            new_row[text_column] = augmented_text
            augmented_rows.append(new_row)

    if len(augmented_rows) == 0:
        return train_df.copy(), 0

    augmented_df = pd.DataFrame(augmented_rows)

    combined_df = pd.concat(
        [train_df, augmented_df],
        ignore_index=True
    )

    return combined_df, len(augmented_rows)
