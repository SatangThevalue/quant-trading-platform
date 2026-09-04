import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from loguru import logger

class ForexNewsSentiment:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Using a reliable financial RSS feed (ForexLive/Investing as proxies, here using Yahoo Finance for forex as example)
        # In production, a premium API like FinancialJuice or ForexFactory JSON is preferred.
        self.rss_urls = {
            "USD": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X,GBPUSD=X",
            "GOLD": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F"
        }

    def fetch_and_analyze(self, asset="USD"):
        url = self.rss_urls.get(asset)
        if not url: return 0.0

        try:
            feed = feedparser.parse(url)
            if not feed.entries: return 0.0

            total_score = 0
            count = 0
            
            # Analyze top 5 most recent news headlines
            for entry in feed.entries[:5]:
                title = entry.title
                # Vader gives a compound score between -1 (most extreme negative) and +1 (most extreme positive)
                score = self.analyzer.polarity_scores(title)['compound']
                total_score += score
                count += 1
                logger.info(f"[NLP] Headline: '{title}' | Score: {score}")

            avg_score = total_score / count if count > 0 else 0
            logger.info(f"[NLP] Average Sentiment for {asset}: {avg_score:.2f}")
            return avg_score
            
        except Exception as e:
            logger.error(f"NLP Error: {e}")
            return 0.0

if __name__ == "__main__":
    nlp = ForexNewsSentiment()
    print("--- USD News Sentiment ---")
    nlp.fetch_and_analyze("USD")
    print("\n--- GOLD News Sentiment ---")
    nlp.fetch_and_analyze("GOLD")
