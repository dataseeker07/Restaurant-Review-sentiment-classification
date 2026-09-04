import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model=joblib.load('Restaurant_review_model.pkl')
vectorizer=joblib.load('tfidf_vectorizer.pkl')

ps=PorterStemmer()

custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
            'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
            'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

stop_words=set(stopwords.words("english"))-custom_stopwords

def preprocess_text(text):
            text=re.sub('[^a-zA-Z]',' ',text)
            text=text.lower()
            words=text.split()
            words=[ps.stem(word) for word in words if word not in stop_words]
            return " ".join(words)

st.title("🍽️Restaurant Review Classification🍽️")

review=st.text_area("Enter your restuarant review:")

if st.button("Classify"):
    if review:
        processed=preprocess_text(review)
        vectorized=vectorizer.transform([processed])
        prediction=model.predict(vectorized)[0]
        sentiment = "Positive 😊" if prediction== 1 else "Negative 😔"
        st.subheader("Predicted Sentiment")
        st.success(sentiment)
    else:
        st.warning("Please enter a review.")
        