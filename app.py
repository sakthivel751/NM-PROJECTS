import pandas as pd
import re
import string
import plotly.graph_objs as go
from flask import Flask, render_template
from textblob import TextBlob
from collections import Counter
import base64
import io
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Sentiment analysis function
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/')
def index():
    # Load dataset
    url = 'https://raw.githubusercontent.com/Jagathish67/Decoding-emotions-through-sentiment-analysis-of-social-media-conversations/main/submission.csv'
    train = pd.read_csv(url)
    train.dropna(inplace=True)

    # Clean text columns if available
    if 'text' in train.columns:
        train['text'] = train['text'].apply(clean_text)
    train['selected_text'] = train['selected_text'].apply(clean_text)

    # Calculate sentiment
    train['sentiment'] = train['selected_text'].apply(get_sentiment)

    # Sentiment distribution plot (Seaborn)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=train, palette='pastel')
    plt.title('Sentiment Distribution')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Top 10 selected texts
    top_selected = train['selected_text'].value_counts().head(10)

    # Most common words in selected_text
    train['temp_list'] = train['selected_text'].apply(lambda x: str(x).split())
    top_words_counter = Counter([item for sublist in train['temp_list'] for item in sublist])
    top_words_df = pd.DataFrame(top_words_counter.most_common(10), columns=['Word', 'Count'])
    top_words_list = top_words_df.to_dict(orient='records')

    # Create Funnel Chart (Plotly) for Top 10 Selected Texts
    funnel_fig = go.Figure(go.Funnelarea(
        text=top_selected.index,
        values=top_selected.values,
        title={"position": "top center", "text": "Funnel-Chart of Top Selected Texts"}
    ))

    # Save Funnel chart as base64 image
    funnel_img = io.BytesIO()
    funnel_fig.write_image(funnel_img, format='png')
    funnel_img.seek(0)
    funnel_plot_url = base64.b64encode(funnel_img.getvalue()).decode()

    return render_template('index.html',
                           sentiment_counts=train['sentiment'].value_counts().to_dict(),
                           plot_url=plot_url,
                           top_selected=top_selected.to_dict(),
                           top_words=top_words_list,
                           funnel_plot_url=funnel_plot_url)

if __name__ == '__main__':
    app.run(debug=True)
