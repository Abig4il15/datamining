from multiprocessing.resource_sharer import stop
from time import time
from matplotlib import text
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
import streamlit as st
import numpy as np
import re
import pandas as pd
import string
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
st.title('ANALISIS SENTIMEN KOMENTAR MASYARAKAT TERHADAP PROGRAM MAKAN BERGIZI GRATIS (MBG) DI APLIKASI X ')
st.write("""Metode Support Vector Machine""")
dataset=st.file_uploader("Upload File Excel",type=["csv"])
if dataset:
    df=pd.read_csv(dataset)    
if dataset is not None:
    kolom_dihapus = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    kolom_ditahan = [col for i, col in enumerate(df.columns) if i not in kolom_dihapus]
    df = df[kolom_ditahan]
if dataset is not None:
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        df = df.dropna()        
if dataset is not None:
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
from deep_translator import GoogleTranslator
if dataset is not None:
    def translate_text(text):
        if pd.isna(text):
            return text
        try:
            translated = GoogleTranslator(source='auto', target='id').translate(text)
            return translated
        except Exception as e:
            return text
    df['translate_id'] = df['full_text'].apply(translate_text)
    df=df.drop(columns=['full_text'])

if dataset is not None:
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        df = df.dropna()
if dataset is not None:
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()

if dataset is not None:
    def clean_text(text):
        text = str(text)
        text = re.sub(r'http\S+', '', text)  # hapus link
        text = re.sub(r'@\w+', '', text)     # hapus mention
        text = re.sub(r'#\w+', '', text)     # hapus hashtag
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # hapus angka & simbol
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    df['cleaned_text'] = df['translate_id'].apply(clean_text)
    df=df.drop(columns=['translate_id'])    

if dataset is not None:
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        df = df.dropna()

if dataset is not None:
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()

keywords = [
    'mbg', 'makan bergizi', 'gizi', 'program makan','sekolah','makan','bergizi','gratis','anggaran','program'
    'makan gratis', 'bergizi gratis', 'anak sekolah', 'pemerintah','makangratis','bergizigratis','nutrisi','badangizi','sekolah','dapurumum'
]
if dataset is not None:
    def is_relevant(text):
        if pd.isna(text):
            return False
        text = text.lower()
        return any(keyword in text for keyword in keywords)
    df['is_relevant'] = df['cleaned_text'].apply(is_relevant)
    df = df[df['is_relevant']].copy()

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
stopwords = list(stopwords) + ['dpt','GA','BGN','pegega','awkarin','tau','sbg','gue','rp','trs','pa','doi','ngga','nya', 'banget', 'sih',
                               'dong','loh','icw','jir', 'aja','gweh', 'yg','di','dan','yang','itu','wowo', 'ya', 'ini', 'lu','wkwkwk','yg','oh','dlm','t','gak','kait','ga','gw','dgn','tuh',
                               'kakaa','gua','kkg','Aja','Ga','nih','krn','dlmsolo','dlm','PANTEK','tp','skrg','gw ','stp','BGN','bgt','jule','org','lah','jg','iya','fm',
                               'ku','jin','ta','iki','AJA','cuy','tmpt','jt','Rp','rp','T','cie','jiro','bgttt','pliss','Noh','mah','Iki','Per','Ya','fnb','Kek','sr','vpn','kw','amp','si','ni','t','wok','eh','Mbg','AN','kk','dr','jd','aa','kek','gin','udh','ala','ah','bgt','dg','deh','gatau','babi',
                               'aurelia','sm','mks','blm','ANJIRK','GRILL','bojo','lgi','lbh','hindayana','dari','nntn','udh','laen','bjir','oh','ya','my','kocak','anjing','heh','ye','lu','aduh','mbg','MBG','makan','gizi','sekolah','guru','pemerintah','program',
                               'tu','jokodok','juuussss','Dan','Lah','ppppp','elah','m','gini','Gin','bray','dg','YE','Yg','gBLKKKKKK','cok','doi','gua','ga','tuh','buzzer','btw','YA','NEK','SING','WOWOK','RA','ANA','bestieeee','aseennng','wkwk','yap','kono']

def remove_stopwords(text):
    words = str(text).split()
    filtered = [w for w in words if w not in stopwords]
    return " ".join(filtered)
if dataset is not None:
    df['Stopword'] = df['cleaned_text'].apply(remove_stopwords)
    df=df.drop(columns=['cleaned_text','is_relevant'])

if dataset is not None:
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        df = df.dropna()

if dataset is not None:
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()

if dataset is not None:
    Transform_cases = df.Stopword.str.lower()
    df['Transform_cases'] = df['Stopword'].str.lower()
    df=df.drop(columns=['Stopword'])

if dataset is not None:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    def predict_sentiment(text):
        # Hindari NaN / float / kosong
        if pd.isna(text):
            return "Netral"
        text = str(text).strip()
        if text == "":
            return "Netral"
        # Tokenisasi
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        # Prediksi
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(dim=1).item()
        if predicted_class == 0:
            return "Negative"
        elif predicted_class == 1:
            return "Netral"
        else:
            return "Positive"
    df['Sentiment'] = df['Transform_cases'].apply(predict_sentiment)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
if dataset is not None:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df['stemming'] = df['Transform_cases'].apply(lambda x: stemmer.stem(str(x)))
    df=df.drop(columns=['Transform_cases'])
if dataset is not None:
    df['Tokenizing'] = df['stemming'].apply(lambda x: str(x).split())
    df=df.drop(columns=['stemming'])
    st.dataframe(df)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
if dataset is not None:
    X = df['Tokenizing'].apply(lambda x: ' '.join(x))
    y = df['Sentiment']
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1,
        random_state=42,
        stratify=y_temp
        )
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_tfidf,
        y_train
        )
    model1 = SVC(
        kernel='linear',
        C=1
        )
    model1.fit(X_train_smote, y_train_smote)
    y_val_pred = model1.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    st.write(f"Validation Accuracy: {val_accuracy:.4f}")
    y_test_pred = model1.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    st.write(f"Test Accuracy: {test_accuracy:.4f}")
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
if dataset is not None:
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('smote', SMOTE(random_state=42)),
        ('svc', SVC(kernel='linear', C=1))
    ])
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Average CV Accuracy: {cv_scores.mean():.4f}")
