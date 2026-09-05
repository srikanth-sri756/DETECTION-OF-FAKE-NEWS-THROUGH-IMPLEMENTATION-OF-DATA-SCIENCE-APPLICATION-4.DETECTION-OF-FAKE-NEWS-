# Detection of Fake News Through Implementation of Data Science Application

A desktop data-science application that classifies news/tweet text as **GENUINE** or **FAKE** using a TF-IDF feature pipeline and an LSTM deep-learning classifier, wrapped in a Tkinter GUI.

## Overview

Misinformation spreads faster than corrections, so automated screening of news text is a practical first line of defence. This project takes a labelled corpus of news items, cleans and vectorises the text, trains a stacked LSTM network, and then lets you score fresh, unseen news items from a CSV file.

The whole workflow — dataset upload, preprocessing, training, evaluation and prediction — is driven from a single Tkinter window, so no notebook or command-line experience is required to demo it.

## Features

- **Dataset upload** — load a labelled CSV (`text`, `target` columns) from the `TwitterNewsData/` folder.
- **Text preprocessing** — punctuation stripping, non-alphabetic token removal, English stop-word removal and WordNet lemmatisation.
- **Feature extraction** — TF-IDF vectorisation with unigrams + bigrams, capped at 200 features, followed by L2 normalisation.
- **LSTM classifier** — two stacked LSTM layers (128 units each) with dropout, a 32-unit dense layer and a softmax output.
- **Model caching** — a trained model is persisted to `model/` and reloaded on the next run instead of retraining.
- **Accuracy & loss graph** — per-epoch training curves plotted with Matplotlib.
- **Batch prediction** — score a file of unlabelled news items and see the verdict for each one.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.7 |
| GUI | Tkinter |
| Deep learning | Keras / TensorFlow |
| Classical ML | scikit-learn (TF-IDF, label encoding, train/test split) |
| NLP | NLTK (stop-words, WordNet lemmatiser) |
| Data & plots | pandas, NumPy, Matplotlib |

## Project structure

```
.
├── Main.py                 # Tkinter application: UI + full ML pipeline
├── run.bat                 # Convenience launcher (python Main.py)
├── model/
│   ├── model.json          # Serialised LSTM architecture
│   ├── model_weights.h5    # Trained weights
│   ├── model.txt           # Pickled classifier
│   └── history.pckl        # Per-epoch accuracy/loss history
└── TwitterNewsData/
    ├── news.csv            # Labelled training corpus (text, target)
    └── testNews.txt        # Unlabelled samples for prediction
```

## Getting started

### Prerequisites

- Python 3.7 (the Keras/TensorFlow APIs used here target the TF 1.x / early 2.x era)

### Installation

```bash
git clone https://github.com/srikanth-sri756/.DETECTION-OF-FAKE-NEWS-THROUGH-IMPLEMENTATION-OF-DATA-SCIENCE-APPLICATION-4.DETECTION-OF-FAKE-NEWS-.git
cd .DETECTION-OF-FAKE-NEWS-THROUGH-IMPLEMENTATION-OF-DATA-SCIENCE-APPLICATION-4.DETECTION-OF-FAKE-NEWS-

pip install pandas numpy matplotlib scikit-learn nltk keras tensorflow
```

Download the NLTK corpora the preprocessing step depends on:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running

```bash
python Main.py
```

or double-click `run.bat` on Windows.

## Usage

1. **Upload Dataset** — pick `TwitterNewsData/news.csv`. Cleaned text and labels are echoed into the output pane.
2. **Preprocess** — builds the TF-IDF matrix and performs an 80/20 train/test split.
3. **Run LSTM** — trains the network for 10 epochs, or loads the cached model from `model/` if one exists, then reports accuracy.
4. **Graph** — plots accuracy and loss against epochs.
5. **Predict** — choose a test file; each news item is printed with a **GENUINE** or **FAKE** verdict.

> Run the steps in order — preprocessing populates the vectoriser and tensors that training and prediction rely on.

## Notes

- `Main.py` uses `DataFrame.get_value()`, which was removed in pandas 1.0. Either pin an older pandas release or swap those calls for `.at[]` / `.iat[]` if you run on a modern stack.
- Accuracy is read from epoch 10 of the saved history, so the training loop expects at least 10 epochs.
- Delete the contents of `model/` to force a full retrain.

## License

Released for academic and educational use.
