# Aspect-Based Sentiment Analysis (ABSA)

Aspect-based sentiment analysis predicts the sentiment (negative / positive / neutral) expressed towards a **specific aspect** in a sentence, e.g. *"The battery life is great but the screen is dim"* → `battery life`: positive, `screen`: negative.

This project compares classical baselines against several BERT-family architectures on the SemEval ATSA laptop and restaurant datasets.

## Repository layout

```
data/
  raw/                  Raw laptop training CSV
  processed/            Preprocessed CSV (aspect_context, label)
src/
  Data/                 SemEval ATSA train/test JSON (atsa-laptop, atsa-restaurant)
  preprocessing/
    preprocess.ipynb    Data cleaning and preparation
  models/
    tfidf_naive_bayes.ipynb       TF-IDF + Multinomial Naive Bayes (manual implementation)
    logistic_regression.ipynb     TF-IDF + softmax regression (manual implementation)
    bert_finetune.ipynb           Fine-tuned BertForSequenceClassification with evaluation metrics
    BERT_ABSA_KFold.ipynb         BERT / BERT+LSTM / BERT+Attention with 10-fold cross-validation
    BERT_ABSA_LSTM_ATTN.ipynb     Adds BERT + BiLSTM + token attention, 5-fold CV, QLoRA experiment
    Clean_BERT.ipynb              Cleaned-up K-fold pipeline for all four BERT variants
    Different_Bert.ipynb          Compares bert-base-uncased, roberta-base and deberta-base
    Final Model.ipynb             Final consolidated pipeline: training, checkpoints, inference, comparison
```

## Approaches

| Family | Models |
| --- | --- |
| Baselines | TF-IDF + Naive Bayes, TF-IDF + softmax regression |
| BERT variants | BERT base classifier, BERT + LSTM, BERT + layer attention, BERT + BiLSTM + token attention |
| Other encoders | RoBERTa-base, DeBERTa-base |
| Parameter-efficient | QLoRA (4-bit + LoRA) on BERT base |

Sentence and aspect are encoded as a sentence pair; stratified K-fold cross-validation is used to get robust estimates on the small datasets. Metrics reported are accuracy, macro F1, per-class classification reports and confusion matrices.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn jupyter
# optional, for QLoRA
pip install peft bitsandbytes accelerate
jupyter notebook
```

Run `src/preprocessing/preprocess.ipynb` first for the classical baselines, then any notebook in `src/models/`. A GPU is strongly recommended for the BERT notebooks. Trained checkpoints (`*.pth`) are written locally and are git-ignored.

## Datasets

SemEval ATSA (aspect term sentiment analysis) laptop and restaurant reviews, stored under `src/Data/`. The laptop training set is also provided as CSV under `data/`.
