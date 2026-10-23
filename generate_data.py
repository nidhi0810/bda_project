"""Optional fallback generator. The primary project dataset should be the Kaggle CSV.
This creates synthetic data only for testing the UI if the Kaggle file is unavailable.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 1500
    topics = rng.choice(["dance", "technology", "music", "food", "fashion", "gaming"], n)
    platforms = rng.choice(["Instagram", "TikTok", "YouTube", "X"], n)
    sentiments = rng.choice(["Positive", "Neutral", "Negative"], n, p=[.48, .32, .20])
    views = rng.lognormal(7, 1, n).astype(int)
    likes = (views * rng.uniform(.01, .09, n)).astype(int)
    shares = (views * rng.uniform(.001, .02, n)).astype(int)
    comments = (views * rng.uniform(.001, .015, n)).astype(int)
    df = pd.DataFrame({
        "post_id": [f"SYN{i:05d}" for i in range(n)],
        "platform": platforms,
        "hashtag": ["#" + x for x in topics],
        "topic": topics,
        "sentiment": sentiments,
        "region": rng.choice(["India", "US", "UK", "Other"], n),
        "content_type": rng.choice(["video", "image", "text", "reel"], n),
        "timestamp": (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")).astype(str),
        "views": views, "likes": likes, "shares": shares, "comments": comments
    })
    path = Path("data/multi_platform_social_sentiment_evolution.csv")
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} synthetic rows at {path}. This is NOT the Kaggle dataset.")

if __name__ == "__main__":
    main()
