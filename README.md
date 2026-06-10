# 🌐 Multilingual Language Detection & Translation

An end-to-end NLP pipeline that automatically **detects the language** of any input text and **translates it** into a target language — supporting 15 languages out of the box.

---

## 📌 Overview

This project combines a classical machine learning classifier with a state-of-the-art neural translation model:

- **Language Detection** — A TF-IDF + Logistic Regression model trained on a balanced multilingual corpus.
- **Machine Translation** — Facebook's [NLLB-200 (1.3B)](https://huggingface.co/facebook/nllb-200-1.3B) model for high-quality, multilingual neural translation.

The pipeline takes raw text, preprocesses it, predicts its language automatically, then translates it to any supported target language chosen by the user.

---

## 🧠 Supported Languages

| # | Language | NLLB Code |
|---|----------|-----------|
| 1 | English | `eng_Latn` |
| 2 | Arabic | `arb_Arab` |
| 3 | French | `fra_Latn` |
| 4 | Spanish | `spa_Latn` |
| 5 | Portuguese | `por_Latn` |
| 6 | German | `deu_Latn` |
| 7 | Italian | `ita_Latn` |
| 8 | Dutch | `nld_Latn` |
| 9 | Turkish | `tur_Latn` |
| 10 | Russian | `rus_Cyrl` |
| 11 | Chinese | `zho_Hans` |
| 12 | Japanese | `jpn_Jpan` |
| 13 | Korean | `kor_Hang` |
| 14 | Hindi | `hin_Deva` |
| 15 | Swedish | `swe_Latn` |

---

## 🗂️ Project Structure

```
├── NLP_Project.ipynb       # Main notebook (EDA → Training → Translation)
├── Language Detection.csv  # Dataset (mounted from Google Drive)
└── README.md
```

---

## ⚙️ Pipeline

```
Raw Text
   │
   ▼
Preprocessing (tokenization, POS-aware lemmatization, punctuation removal)
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Logistic Regression → Detected Language
   │
   ▼
NLLB-200 Neural Translation → Translated Output
```

### Text Preprocessing

- Punctuation removal
- Word tokenization via NLTK
- POS tagging for context-aware lemmatization using `WordNetLemmatizer`

### Class Balancing

The dataset is upsampled per language class to a uniform size of **1,385 samples** using `sklearn.utils.resample` to address class imbalance before training.

---

## 📊 Model Performance

The Logistic Regression classifier is evaluated on both training and test splits using:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn nltk tensorflow joblib transformers sentencepiece
```

Download required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```

### Running the Notebook

1. **Mount Google Drive** (if using Colab) and place `Language Detection.csv` in your Drive.
2. Open `NLP_Project.ipynb` in Google Colab or Jupyter.
3. Run all cells in order.
4. When prompted, enter any text — the model will detect its language and ask you to choose a target language for translation.

### Example

```
📝 Enter text to translate:
> क्या आप अंग्रेज़ी बोलते हैं

🧠 Detected source language: Hindi

🎯 Target language (translate TO):
    1. English
    2. Arabic
    ...

> 1

=======================================================
  🔤 Original  (Hindi): क्या आप अंग्रेज़ी बोलते हैं
  ✅ Translated (English): Do you speak English?
=======================================================
```

---

## 🔧 Tech Stack

| Component | Library / Model |
|-----------|----------------|
| Data handling | `pandas`, `numpy` |
| Text preprocessing | `nltk` (tokenizer, lemmatizer, POS tagger) |
| Feature extraction | `sklearn` TF-IDF Vectorizer |
| Language classifier | `sklearn` Logistic Regression |
| Neural translation | `facebook/nllb-200-1.3B` via HuggingFace Transformers |
| Model persistence | `joblib` |

---

## 📄 Dataset

The model is trained on the **Language Detection** dataset, a multilingual text corpus with labeled language samples spanning 15 languages.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
