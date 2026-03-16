# News Sentiment Pipeline

A Python tool that fetches news via NewsAPI and analyzes headline sentiment using Hugging Face's AI models.

## Setup

1. Install required libraries:
   pip install pandas requests matplotlib huggingface_hub python-dotenv

2. Create a .env file in this folder and add your keys:
   NEWS_API_KEY=your_key_here
   HF_TOKEN=your_token_here

## How it Works

1. Fetch: Downloads the 30 latest news articles on "Artificial Intelligence."
2. Analyze: Uses a RoBERTa model to label headlines as Positive, Negative, or Neutral.
3. Visualize: Displays a bar chart of sentiments and a confidence histogram.

## Running the Code

Simply run:
python main.py

## Project Structure
- main.py: The Python script.
- .env: Your private API keys (do not share).
- README.md: Project documentation.
