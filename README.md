# 🎬 Movie Recommendation System

A concise content-based movie recommendation system using Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity over movie metadata (overview, genres, tagline).

---

## Overview

This repository demonstrates a content-based recommender that finds movies similar to a given title by comparing text-based features. It is designed for easy experimentation and a simple Streamlit demo.

---

## Features

- Text preprocessing (tokenization, stop-word removal, lemmatization).
- Combined metadata representation (`overview`, `genres`, `tagline`) as searchable `tags`.
- TF-IDF vectorization with configurable n-gram range and feature size.
- Cosine similarity for fast nearest-neighbor lookups.
- Optional Streamlit interface for interactive exploration.

---

## Repository Structure

```text
Movie recommendation system/
├── app.py                             # Optional Streamlit app
├── Movie_recommendation_system.ipynb  # Notebook: EDA, preprocessing, model building
├── tfidf.pkl                          # (optional) saved TfidfVectorizer
├── tfidf_matrix.pkl                   # (optional) saved TF-IDF matrix
├── df.pkl                             # (optional) processed DataFrame
├── indices.pkl                        # (optional) title -> index mapping
└── README.md                          # This file
```

---

## Requirements

- Python 3.8+
- streamlit (optional)
- pandas, numpy
- scikit-learn
- nltk (for lemmatization)

You can install the common dependencies with:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install minimal packages directly:

```bash
pip install streamlit scikit-learn pandas nltk
```

---

## Quick Start

1. If using the Streamlit demo (optional):

```bash
streamlit run app.py
```

2. Or use the model programmatically by loading the pickled artifacts (if present):

```python
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load artifacts (if you have them)
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
indices = pickle.load(open('indices.pkl', 'rb'))
df = pd.read_pickle('df.pkl')

def recommend(title, n=5):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_indices].tolist()

print(recommend('The Transporter', n=5))
```

---

## Dataset

This project expects cleaned movie metadata (title, overview, genres, tagline). The notebook `Movie_recommendation_system.ipynb` contains the data-prep and TF-IDF pipeline used to generate the pickled artifacts.

---

## Contributing

Contributions are welcome. Open an issue or submit a pull request with clear intent and tests/examples.

---

## License & Contact

This project is provided for educational purposes. Add a license file if you plan to publish or share commercially. For questions, contact the repository owner.

