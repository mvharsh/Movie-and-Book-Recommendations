# 🎥📚 **Movie and Book Recommendations with Genre Mapping and Twitter Sentiment Analysis**

This project provides an intelligent, mood-aware content discovery tool that analyzes Twitter-like text input and recommends movies and books based on detected sentiment. The system utilizes a RoBERTa-based sentiment analyzer and a genre mapping engine to connect emotional tone with personalized media suggestions.

---

## 🚀 Features

- **RoBERTa Twitter Sentiment Analyzer**
  - Detects Positive, Neutral, and Negative sentiments
  - Outputs sentiment probabilities and dominant sentiment

- **Genre Mapping System**
  - Maps sentiment → genres
  - E.g., Positive → Comedy, Romance; Negative → Horror, Psychological

- **Media Recommendation Engine**
  - Recommends mock book/movie data based on sentiment
  - Includes title, author/director, genre, year, and description

- **Interactive Streamlit Interface**
  - Analyze a single text input and get instant media suggestions
  - Upload CSV for batch analysis and visualize sentiment insights
  - Explore genre mapping logic

---

## 🖥️ Application Preview

![App Preview](https://user-images.githubusercontent.com/your-username/app-preview.png) <!-- Optional if you have a screenshot -->

---

## 📂 Project Structure

```
📁 project-root/
│
├── app.py              # Streamlit app
├── sentiment.py        # Sentiment analysis and recommendation logic
├── requirements.txt    # Python dependencies
└── README.md           # Project overview

```
---

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/mvharsh/Movie-and-Book-Recommendations-with-Genre-Mapping-Twitter-Sentiment-Analysis.git
cd Movie-and-Book-Recommendations-with-Genre-Mapping-Twitter-Sentiment-Analysis
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the application**:

```bash
streamlit run app.py
```

---

## 🔍 Key Components

### 1. Sentiment Analysis
- Uses `cardiffnlp/twitter-roberta-base-sentiment`
- Preprocesses input tweets for user mentions and links
- Outputs probabilities for Positive, Neutral, and Negative

### 2. Genre Mapping
- Sentiments are associated with emotional genres:
    - **Positive**: Comedy, Romance, Family, Animation...
    - **Neutral**: Sci-Fi, Documentary, History...
    - **Negative**: Horror, Psychological, Thriller...

### 3. Recommendations
- **Mock APIs** simulate data from TMDb and OpenLibrary
- Recommendations based on:
    - Dominant sentiment
    - Weighted sentiment profile

---

## 📊 Streamlit App Features

| Feature                    | Description |
|---------------------------|-------------|
| **Single Text Analysis**  | Analyze one tweet or post and get media recs |
| **Batch CSV Analysis**    | Upload CSV of tweets, analyze all at once, download results |
| **Genre Mapping Explorer**| Understand how emotions map to genres |

---

## 📈 Visualizations

- Sentiment Probability Bar Charts
- Pie Charts for batch sentiment distribution
- Word Cloud per sentiment
- Stacked Genre Frequency Charts

---

## 📁 Example CSV for Batch Analysis

```csv
id,text
1,"I just had an amazing coffee and I feel great!"
2,"The weather is so gloomy, it's making me sad."
3,"Watched a really thought-provoking documentary today."
```

---

## ▶️ Youtube Link

https://www.youtube.com/watch?v=zh6AuxX8JVQ

