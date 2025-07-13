# 🎬 TMDb Movie Recommendation System

A content-based movie recommender system built using the **TMDb 5000 Movie Dataset**. It uses natural language processing (NLP) and machine learning techniques to suggest similar movies based on metadata like genres, keywords, cast, and crew.

---

## 📁 Dataset

This project uses the [TMDb 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), which contains two CSV files:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

The dataset includes metadata such as genres, cast, crew, keywords, and overviews.

---

## 💡 Features

- ✔️ Parses and processes JSON-style strings in the dataset
- ✔️ Extracts top 3 cast members and director for each movie
- ✔️ Combines multiple fields into a single `tags` field
- ✔️ Vectorizes text using **CountVectorizer**
- ✔️ Calculates **cosine similarity** between movies
- ✔️ Recommends top 5 similar movies
- ✔️ Saves preprocessed data and similarity matrix using `pickle`

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- scikit-learn
- Natural Language Processing
- Cosine Similarity
- CountVectorizer
- PyPi Libraries: `ast`, `pickle`, `io`, `zipfile`, `os`

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
