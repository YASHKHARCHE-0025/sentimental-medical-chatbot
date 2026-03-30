# sentimental-medical-chatbot
A machine learning-based NLP project that classifies user input (symptoms/statements) into predefined categories using TF-IDF + Logistic Regression / SVM.


** Overview **

This project builds a text classification system (chatbot-like) that predicts a category (e.g., disease/status) based on user input.

>>The system:

--Cleans and preprocesses text
--Converts text into numerical features using TF-IDF
--Trains ML models to classify input
--Predicts output for new user queries

>>Features
-- NLP preprocessing (tokenization, stopword removal, lemmatization)
-- TF-IDF vectorization (unigram + bigram)
-- Multiple ML models:
===Logistic Regression==
===Support Vector Machine (LinearSVC)===
---Model evaluation (accuracy + cross-validation)
---Real-time prediction using user input
---Handles imbalanced data (class_weight='balanced')


>>Tech Stack
  --Python 
  --Pandas & NumPy
  --NLTK
  --Scikit-learn
  --Imbalanced-learn
