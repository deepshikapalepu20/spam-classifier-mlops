import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle, urllib.request
 
# Download the SMS Spam dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df  = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
 
X = df['message']
y = df['label'].map({'ham': 0, 'spam': 1})
 
# Build pipeline: TF-IDF vectorizer + Naive Bayes
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf',   MultinomialNB()),
])
model.fit(X, y)
 
# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
 
print("Model trained and saved as model.pkl")
print(f"Training samples: {len(X)}")
