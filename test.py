import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('twitter_training.csv')
data.columns = ['id', 'category', 'sentiment', 'text']

data = data.dropna(subset=['text'])

data['text'] = data['text'].fillna('') 

data = data[['sentiment', 'text']]
data = data[data['sentiment'].isin(['Positive', 'Neutral', 'Negative'])]

def clean_text(text):
    if isinstance(text, str):  
        text = text.lower()  
        text = re.sub(r'[^a-zA-Z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text)  
        return text.strip() 
    else:
        return ''  

data['cleaned_text'] = data['text'].apply(clean_text)

sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
data['sentiment_encoded'] = data['sentiment'].map(sentiment_mapping)

X = data['cleaned_text']
y = data['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)  
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200, solver='liblinear')  
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
