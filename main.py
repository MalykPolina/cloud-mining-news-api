import pandas as pd
import requests
import matplotlib.pyplot as plt
import time
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
QUERY = "artificial intelligence"

if not NEWS_API_KEY or not HF_TOKEN:
    print("Error: API keys not found in .env file!")

def fetch_news_data(query):
    print(f"Fetching news for: {query}...")
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=30&apiKey={NEWS_API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: NewsAPI returned status {response.status_code}")
        return pd.DataFrame()

    articles = response.json().get('articles', [])
    data = []
    for art in articles:
        data.append({
            "title": art['title'],
            "source": art['source']['name'],
            "published_at": art['publishedAt'],
            "url": art['url']
        })

    df = pd.DataFrame(data)
    print(f"Successfully fetched {len(df)} articles.")
    return df


def analyze_sentiment(df):
    client = InferenceClient(token=HF_TOKEN)
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    results_labels = []
    results_scores = []

    print("Running sentiment analysis via HF Router...")
    for index, row in df.iterrows():
        try:
            prediction = client.text_classification(row['title'], model=model_id)

            top_result = max(prediction, key=lambda x: x['score'])

            results_labels.append(top_result['label'])
            results_scores.append(top_result['score'])
        except Exception as e:
            print(f"Skipping article {index} due to error: {e}")
            results_labels.append("UNKNOWN")
            results_scores.append(0.0)

        time.sleep(0.2)

    df['sentiment'] = results_labels
    df['confidence'] = results_scores
    return df


def create_visuals(df):
    if df.empty: return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    df['sentiment'].value_counts().plot(kind='bar', color=['#4CAF50', '#F44336', '#9E9E9E'])
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(df['confidence'], bins=10, color='skyblue', edgecolor='black')
    plt.title("Model Confidence Scores")
    plt.xlabel("Score")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_news = fetch_news_data(QUERY)

    if not df_news.empty:
        df_analyzed = analyze_sentiment(df_news)

        print("\n--- Final Data Preview ---")
        print(df_analyzed[['title', 'sentiment', 'confidence']].head())

        create_visuals(df_analyzed)