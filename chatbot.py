
import pandas as pd
from string import punctuation, digits
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

dataset = pd.read_csv(r"E:\dataset\chatbot dataset\Combined Data.csv")

dataset.head(5)


if 'Unnamed: 0' in dataset.columns:
    dataset.drop(columns=['Unnamed: 0'], inplace=True)
dataset.dropna(subset=['statement', 'status'], inplace=True)

dataset = dataset.drop_duplicates()
dataset = dataset.dropna(subset=['statement', 'status'])
dataset = dataset.reset_index(drop=True)


def remove_extra(data):
    string = ""
    for i in data:
        if i not in punctuation and i not in digits:
            string +=i
    return string


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def remove_stop_lemmatize(text):
    text = text.lower()
    words = word_tokenize(text)
    text = text.lower()
    new_words = []

    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            new_words.append(lemma)

    return " ".join(new_words)


dataset["statement"] = dataset["statement"].astype(str).apply(remove_stop_lemmatize)
dataset['statement'] = dataset['statement'].apply (remove_extra)
dataset = dataset[dataset["statement"].str.strip() != ""]
dataset['status']=dataset['status'].str.lower()
df = dataset.copy()


le = LabelEncoder()
y = le.fit_transform(df['status'])


vector = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.85)
x = vector.fit_transform(df['statement'])


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=44,shuffle=True)


model = LogisticRegression(C=1,max_iter=2000,class_weight='balanced')
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred)*100)

print("Train Accuracy:", model.score(x_train, y_train)*100)
print("Test Accuracy:", model.score(x_test, y_test)*100)


scores = cross_val_score(model, x_train, y_train, cv=2)
print("CV Scores:", scores*100)
print("Mean Accuracy:", scores.mean()*100)


model2 = LinearSVC(C=0.5,max_iter=2000, class_weight='balanced')
model2.fit(x_train,y_train)


ypred1 = model2.predict(x_test)

accuracy_score(y_test,ypred1)*100

print("Train Accuracy:", model2.score(x_train, y_train)*100)
print("Test Accuracy:", model2.score(x_test, y_test)*100)

scores1 = cross_val_score(model2, x_train, y_train, cv=2)
print("CV Scores:", scores1*100)
print("Mean Accuracy:", scores1.mean()*100)


params = {'C': [0.1,0.2, 0.5, 1]}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    params,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    )

grid.fit(x_train, y_train)

model3 = grid.best_estimator_


y_pred = model3.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred)*100)

print("Train Accuracy:", model3.score(x_train, y_train)*100)
print("Test Accuracy:", model3.score(x_test, y_test)*100)


def identifier(query):
    query = remove_stop_lemmatize(query)
    query = remove_extra(query)

    input_vector = vector.transform([query])
    prediction = model3.predict(input_vector)
    proba = model3.predict_proba(input_vector)

    confidence = np.max(proba)
    if confidence < 0.5:
        print("I am not sure")
    
    label = le.inverse_transform(prediction)
    print("Predicted illness:")
    return label[0]
    


print(identifier(input("Enter your symptoms: ")))




