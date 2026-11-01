# 用于测试异常输入置信度判断和安全拦截逻辑
import os
import re
import copy
import torch
import pandas as pd

from model import build_model
from runtime import (
    FINAL_EXPERIMENT_NAME,
    OUTPUT_DIR,
    configure_quantized_backend,
    load_distilbert_tokenizer,
)


LABEL_MAP = {
    0: "complaint_suggestion",
    1: "feature_feedback",
    2: "problem_consultation"
}


CONFIDENCE_THRESHOLD = 0.60
MIN_ALPHA_CHARS = 5
MIN_WORD_COUNT = 2


DOMAIN_KEYWORDS = {
    "error",
    "crash",
    "crashing",
    "bug",
    "problem",
    "issue",
    "fail",
    "failed",
    "failure",
    "unable",
    "cannot",
    "can't",
    "login",
    "account",
    "app",
    "software",
    "feature",
    "feedback",
    "suggest",
    "suggestion",
    "improve",
    "improvement",
    "slow",
    "performance",
    "support",
    "update",
    "install",
    "payment",
    "page",
    "server",
    "connect",
    "connection",
    "password",
    "question",
    "help"
}


def is_keyboard_smash(text):
    """
    检测明显无意义的键盘乱序输入
    该规则用于在异常输入测试中进行基础过滤
    """

    if text is None:
        return False

    text_lower = str(text).lower()

    keyboard_patterns = [
        "asdf",
        "asdfgh",
        "asdfghjkl",
        "qwerty",
        "qwertyuiop",
        "zxcv",
        "zxcvbnm"
    ]

    for pattern in keyboard_patterns:
        if pattern in text_lower:
            return True

    return False


def has_domain_signal(text):
    """
    检查输入是否至少包含一个与软件反馈、投诉、
    功能请求或问题咨询相关的关键词
    """

    if text is None:
        return False

    text_lower = str(text).lower()

    return any(
        re.search(
            rf"(?<![A-Za-z]){re.escape(keyword)}(?![A-Za-z])",
            text_lower,
        )
        for keyword in DOMAIN_KEYWORDS
    )


def validate_input_text(text):
    """
    在文本送入模型前执行基础输入校验，避免把无意义的异常输入强制归入三个业务类别之一
    """

    if text is None:
        return False, "invalid_input_none"

    text_str = str(text)
    stripped_text = text_str.strip()

    if len(stripped_text) == 0:
        return False, "invalid_input_empty_or_spaces"

    alpha_chars = re.findall(
        r"[A-Za-z]",
        stripped_text
    )

    if len(alpha_chars) < MIN_ALPHA_CHARS:
        return False, "invalid_input_too_few_english_letters"

    words = re.findall(
        r"[A-Za-z]+",
        stripped_text
    )

    if len(words) < MIN_WORD_COUNT:
        return False, "invalid_input_too_few_english_words"

    if is_keyboard_smash(stripped_text):
        return False, "invalid_input_keyboard_smash"

    return True, "passed_basic_validation"


def predict_single_text(
    model,
    tokenizer,
    text,
    device,
    max_length=128
):
    model.eval()

    encoding = tokenizer(
        str(text),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, predicted_label = torch.max(
            probabilities,
            dim=1
        )

    predicted_label_id = int(
        predicted_label.cpu().item()
    )

    confidence_score = float(
        confidence.cpu().item()
    )

    return predicted_label_id, confidence_score


def make_safe_decision(
    input_text,
    predicted_label_name,
    confidence
):
    """
    判断是否接受模型预测，或将输入标记为无效及人工复核状态
    """

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "final_decision": "low_confidence_review",
            "safe_output": "low_confidence_review",
            "decision_reason": (
                f"Model confidence {confidence:.4f} is below "
                f"threshold {CONFIDENCE_THRESHOLD:.2f}."
            )
        }

    if not has_domain_signal(input_text):
        return {
            "final_decision": "out_of_domain_review",
            "safe_output": "out_of_domain_review",
            "decision_reason": (
                "Input passed basic validation but does not contain "
                "clear software-feedback-related domain signal."
            )
        }

    return {
        "final_decision": "accepted_classification",
        "safe_output": predicted_label_name,
        "decision_reason": "Prediction accepted after validation and confidence check."
    }


def main():
    output_dir = OUTPUT_DIR / "abnormal_input_results"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    device = torch.device("cpu")

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "checkpoints",
        FINAL_EXPERIMENT_NAME,
        "best_model.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Please run train_final_model.py first."
        )

    tokenizer = load_distilbert_tokenizer()

    fp32_model = build_model(
        num_labels=3,
        num_hidden_layers=4,
        dropout=0.2,
        attention_dropout=0.2
    )

    fp32_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    fp32_model = fp32_model.to(device)
    fp32_model.eval()

    quantized_backend = configure_quantized_backend()
    print(f"Using quantized backend: {quantized_backend}")

    optimized_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model),
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    optimized_model.eval()

    abnormal_cases = [
        {
            "input_type": "empty_string",
            "input_text": ""
        },
        {
            "input_type": "spaces_only",
            "input_text": "     "
        },
        {
            "input_type": "symbols_only",
            "input_text": "!!!! ???? #### $$$$ %%%%"
        },
        {
            "input_type": "numbers_only",
            "input_text": "1234567890 9876543210"
        },
        {
            "input_type": "random_letters",
            "input_text": "asdfghjkl qwertyuiop zxcvbnm"
        },
        {
            "input_type": "very_long_repeated_text",
            "input_text": "This software keeps crashing and showing errors. " * 100
        },
        {
            "input_type": "non_english_chinese",
            "input_text": "这个软件一直崩溃，我不知道怎么解决。"
        },
        {
            "input_type": "mixed_language",
            "input_text": "The app keeps crashing，而且我无法登录 account."
        },
        {
            "input_type": "irrelevant_text",
            "input_text": "The weather is nice today and I want to eat noodles."
        },
        {
            "input_type": "html_like_text",
            "input_text": "<html><body>Error 404: page not found</body></html>"
        },
        {
            "input_type": "url_like_text",
            "input_text": "https://example.com/support?id=12345"
        },
        {
            "input_type": "none_value",
            "input_text": None
        }
    ]

    results = []

    for case in abnormal_cases:
        input_type = case["input_type"]
        input_text = case["input_text"]

        try:
            is_valid, validation_reason = validate_input_text(
                input_text
            )

            if not is_valid:
                result = {
                    "input_type": input_type,
                    "input_text": str(input_text),
                    "status": "success",
                    "model_called": False,
                    "validation_passed": False,
                    "validation_reason": validation_reason,
                    "predicted_label_id": "",
                    "predicted_label_name": "",
                    "confidence": "",
                    "final_decision": "invalid_input",
                    "safe_output": "invalid_input",
                    "decision_reason": (
                        "Input was rejected before model inference."
                    ),
                    "error_message": ""
                }

            else:
                predicted_label_id, confidence = predict_single_text(
                    model=optimized_model,
                    tokenizer=tokenizer,
                    text=input_text,
                    device=device,
                    max_length=128
                )

                predicted_label_name = LABEL_MAP.get(
                    predicted_label_id,
                    "unknown"
                )

                safe_decision = make_safe_decision(
                    input_text=input_text,
                    predicted_label_name=predicted_label_name,
                    confidence=confidence
                )

                result = {
                    "input_type": input_type,
                    "input_text": str(input_text),
                    "status": "success",
                    "model_called": True,
                    "validation_passed": True,
                    "validation_reason": validation_reason,
                    "predicted_label_id": predicted_label_id,
                    "predicted_label_name": predicted_label_name,
                    "confidence": confidence,
                    "final_decision": safe_decision["final_decision"],
                    "safe_output": safe_decision["safe_output"],
                    "decision_reason": safe_decision["decision_reason"],
                    "error_message": ""
                }

        except Exception as e:
            result = {
                "input_type": input_type,
                "input_text": str(input_text),
                "status": "failed",
                "model_called": "",
                "validation_passed": "",
                "validation_reason": "",
                "predicted_label_id": "",
                "predicted_label_name": "",
                "confidence": "",
                "final_decision": "runtime_error",
                "safe_output": "runtime_error",
                "decision_reason": "",
                "error_message": str(e)
            }

        results.append(result)

        print("\nInput type:")
        print(result["input_type"])

        print("Status:")
        print(result["status"])

        print("Validation passed:")
        print(result["validation_passed"])

        print("Validation reason:")
        print(result["validation_reason"])

        print("Model called:")
        print(result["model_called"])

        print("Raw predicted label:")
        print(result["predicted_label_name"])

        print("Confidence:")
        print(result["confidence"])

        print("Final decision:")
        print(result["final_decision"])

        print("Safe output:")
        print(result["safe_output"])

        print("Decision reason:")
        print(result["decision_reason"])

        if result["error_message"]:
            print("Error:")
            print(result["error_message"])

    results_df = pd.DataFrame(results)

    result_path = os.path.join(
        output_dir,
        "input_safety_results.csv"
    )

    results_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8"
    )

    total_cases = len(results_df)

    success_count = (
        results_df["status"] == "success"
    ).sum()

    failed_count = (
        results_df["status"] == "failed"
    ).sum()

    invalid_input_count = (
        results_df["final_decision"] == "invalid_input"
    ).sum()

    low_confidence_review_count = (
        results_df["final_decision"] == "low_confidence_review"
    ).sum()

    out_of_domain_review_count = (
        results_df["final_decision"] == "out_of_domain_review"
    ).sum()

    accepted_classification_count = (
        results_df["final_decision"] == "accepted_classification"
    ).sum()

    rejected_or_review_count = (
        invalid_input_count
        + low_confidence_review_count
        + out_of_domain_review_count
    )

    summary = {
        "total_cases": total_cases,
        "success_count": success_count,
        "failed_count": failed_count,
        "crash_free": failed_count == 0,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "invalid_input_count": invalid_input_count,
        "low_confidence_review_count": low_confidence_review_count,
        "out_of_domain_review_count": out_of_domain_review_count,
        "accepted_classification_count": accepted_classification_count,
        "rejected_or_review_count": rejected_or_review_count,
        "misclassification_risk_control_enabled": True
    }

    summary_df = pd.DataFrame([summary])

    summary_path = os.path.join(
        output_dir,
        "input_safety_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8"
    )

    print("\n===== Abnormal Input Test Summary with Validation =====")
    print(summary_df)

    print("\nSaved abnormal input results to:")
    print(result_path)

    print("Saved abnormal input summary to:")
    print(summary_path)

    if failed_count == 0:
        print("\nAbnormal input test passed. No crash occurred.")
    else:
        print("\nAbnormal input test found failed cases. Please review the result file.")

    print("\nInput validation and confidence thresholding were applied.")
    print("Invalid or risky abnormal inputs were not blindly accepted as business labels.")


if __name__ == "__main__":
    main()
