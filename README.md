# Lyric and Audio Emotional Alignment as a Predictor of Spotify Track Popularity
### A Feature Engineering Approach

This repository contains the source code, notebooks, documentation, and supporting materials for our undergraduate Computer Science thesis.

The primary objective of this study is to determine whether emotional alignment between song lyrics and audio contributes to predicting Spotify track popularity.

The project explores:

- Lyric emotion analysis using NLP
- Audio emotion analysis from acoustic features
- Emotional alignment feature engineering
- Machine learning models for popularity prediction
- Evaluation of the predictive contribution of emotional alignment

---

## Tech Stack

### Backend
- Python
- FastAPI
- Pandas
- NumPy
- Scikit-learn

### Frontend
- Next.js
- React
- TypeScript

### Data Science
- Jupyter Notebook
- Matplotlib
- Seaborn
- Librosa / Essentia / Spleeter

---

# Project Structure

```
THESIS-PROJECT/
│
├── api/                 # FastAPI backend
├── app/                 # Next.js frontend
├── docs/                # Project documentation
├── notebooks/           # Jupyter notebooks for experimentation
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── requirements.txt     # Python dependencies
└── uv.lock
```

---

### `requirements.txt`
Lists all required Python packages for running the backend and notebooks.

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### Backend Setup

Navigate to the API directory:

```bash
cd api
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Run the FastAPI server:

```bash
fastapi dev
```

or

```bash
uvicorn main:app --reload
```

---

### Frontend Setup

Navigate to the frontend:

```bash
cd app
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

---

## Repository Purpose

This repository serves as the central workspace for:

- Dataset preparation
- Feature engineering experiments
- Machine learning model development
- API implementation
- Frontend application
- Thesis documentation

---

## Researchers

Bachelor of Science in Computer Science

**Thesis Title**

> **Lyric and Audio Emotional Alignment as a Predictor of Spotify Track Popularity: A Feature Engineering Approach**

---

## License

This repository is intended for academic and research purposes.