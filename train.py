
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE, 'data', 'sms_spam_sample.csv')
MODEL_PATH = os.path.join(BASE, 'model.joblib')

df = pd.read_csv(DATA_PATH)
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(CountVectorizer(), MultinomialNB()) #Multinomial Naive Bayes (MNB)
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print('Accuracy:', accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(pipeline, MODEL_PATH) #Trained Machine Learning Model File (saved using joblib)
print('Model saved to', MODEL_PATH)


#train.py = Model বানায় + Save করে
'''এর কাজ:

CSV ডেটা পড়ে

Text → Number (CountVectorizer)

Naive Bayes দিয়ে Train করে

Accuracy দেখায়

শেষের লাইন:
এখানেই model.joblib ফাইল তৈরি হয়।

👉 এই model টাকেই পরে predict.py ও spamdetector.py load করছে।
'''
