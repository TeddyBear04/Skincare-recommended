from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

import nltk
nltk.download('vader_lexicon')

def enrich_sentiment_scores(df):
    sia = SentimentIntensityAnalyzer()

    df['review_sentiment'] = df['review_title'].apply(
        lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )

    product_sentiment_df = df.groupby('product_id').agg({
        'review_sentiment': 'mean',
        'rating': 'mean',
        'is_recommended': 'mean'
    }).rename(columns={
        'review_sentiment': 'avg_sentiment',
        'rating': 'avg_rating',
        'is_recommended': 'recommend_ratio'
    })

    scaler = MinMaxScaler()
    product_sentiment_df[['sentiment_score_norm', 'rating_norm', 'recommend_norm']] = scaler.fit_transform(
        product_sentiment_df[['avg_sentiment', 'avg_rating', 'recommend_ratio']]
    )

    df = df.merge(
        product_sentiment_df[['sentiment_score_norm', 'rating_norm', 'recommend_norm']],
        on='product_id', how='left'
    )
    return df
