# DistilBERT Compression for Feedback Classification

A compact three-class English feedback classifier built with PyTorch and Hugging Face Transformers. The pipeline covers cleaning and deduplication, data partitioning, fine-tuning, structural attention-head pruning, dynamic INT8 quantization, and evaluation.

## Saved experimental results

| Measure | Recorded result |
|---|---:|
| Unique records after cleaning | 350 |
| Train / validation / test examples | 280 / 35 / 35 |
| Parameter count | 66.96M → 51.21M |
| Parameter reduction | 23.52% |
| FP32 / INT8 serialized size | 195.36 / 117.19 MB |
| Test accuracy | 94.29% on 35 examples |
| Test weighted F1 | 0.9389 |

The small test set limits precision and generalization claims. Dynamic INT8 reduced model size but did not accelerate inference on the recorded Apple Silicon/QNNPACK setup. Repeated inference consistency is not an estimate of training-seed variability.

## Contents

- `run_pipeline.py`: preparation, training, quantization, and evaluation stages.
- `model.py`, `train_final_model.py`: model construction, layer reduction, pruning, and training.
- `evaluate_*.py`: independent evaluation, quantization sensitivity, input controls, and inference repeatability.
- `outputs/`: selected aggregate training and evaluation CSVs.
- [Detailed technical report in Chinese](REPORT_ZH.md).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py --help
```

The original business-text dataset and model checkpoints are excluded. The original full-data instructions in `REPORT_ZH.md` describe the local project and are not a promise that those assets are bundled here.

To run the pipeline with data you are authorized to use, provide a UTF-8 CSV at `data/raw/microsoft_text_classification_data.csv`, with `text` and `category` columns. Supported category names are defined in `preprocessing.py`. Then run:

```bash
python run_pipeline.py prepare
python run_pipeline.py train
python run_pipeline.py quantize
python run_pipeline.py test
```

Training obtains the base model separately. Results on replacement data will differ from the saved study. Runtime and performance comparisons must specify hardware, backend, and batch conditions.

`SOURCE_SNAPSHOT.json` records the selected source and saved results as of 2026-11-01.
