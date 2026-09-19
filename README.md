# WWTP EGPCR Transformer

A modularized version of the WWTP fault-detection experiment based on a Transformer model and EGPCR coreset selection.

## Project structure

```text
wwtp-egpcr/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── X_norm1.txt
│   └── X_fault2.txt
├── custom_dataset_artifactsV2/
└── src/
    ├── __init__.py
    ├── config.py
    ├── model.py
    ├── data_utils.py
    ├── metrics.py
    ├── coreset.py
    └── pipeline.py
```

## Install

```bash
pip install -r requirements.txt
```

## Data

Put the following files into `data/`:

- `X_norm1.txt`
- `X_fault2.txt`

## Run

```bash
python main.py
```

Model weights and the fitted scaler are written to `custom_dataset_artifactsV2/`.

