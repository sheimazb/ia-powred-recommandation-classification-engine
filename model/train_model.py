import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load data
df = pd.read_csv('model/data.csv')

# Train TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])

# Save model
joblib.dump(vectorizer, 'model/tfidf_vectorizer.joblib')
joblib.dump(df, 'model/messages_df.joblib')
