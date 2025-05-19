import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import tkinter as tk
from tkinter import filedialog, Text, Scrollbar, Button, Label
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
# Initialize Tkinter
main = tk.Tk()
main.title("Sentiment Analysis Using Machine Learning")
main.geometry("1000x650")
main.config(bg='skyblue')

# Global variables
global df, X, y, X_train, X_test, y_train, y_test, le, tfidf_vectorizer, model
le = LabelEncoder()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Upload dataset
def upload():
    global df
    filename = filedialog.askopenfilename(initialdir="Dataset")
    df = pd.read_csv(filename)
    text.insert(tk.END, filename + ' Loaded\n\n')
    text.insert(tk.END, str(df.shape) + '\n\n')

# Preprocessing
def preprocessing():
    global df, X, y
    global X_resampled, y_resampled
    df['language'] = le.fit_transform(df['language'])
    df['sentiment'] = le.fit_transform(df['sentiment'])
    preprocessed_tweets = [preprocess_text(tweet) for tweet in df['tweet']]
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_tweets)
    tfidf_array = tfidf_matrix.toarray()
    language_reshaped = df['language'].values.reshape(-1, 1)
    X = np.concatenate((tfidf_array, language_reshaped), axis=1)
    y = df['sentiment']
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    text.insert(tk.END, 'Preprocessing complete.\n\n')
    sns.countplot(x=y_resampled)
    plt.show()

# Splitting data
def splitting():
    global X_resampled, y_resampled
    global X, y, X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    text.insert(tk.END, 'Data splitting complete.\n\n')
    text.insert(tk.END, f'X_train: {X_train.shape}, y_train: {y_train.shape}\n')
    text.insert(tk.END, f'X_test: {X_test.shape}, y_test: {y_test.shape}\n\n')

# Train and evaluate models
def train_model(algorithm):
    global X_train, X_test, y_train, y_test
    global model
    model_path = f'model/{algorithm}.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(tk.END, f'{algorithm} model loaded successfully.\n\n')
    else:
        if algorithm == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(max_depth=4)
        elif algorithm == 'RandomForestClassifier':
            model = RandomForestClassifier(n_estimators=40, max_depth=8)
        elif algorithm == 'ExtraTreesClassifier':
            model = ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42)
        elif algorithm == 'RNNClassifier':
            tokenizer = Tokenizer(num_words=5000)
            model = Sequential([
                Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=10),
                SimpleRNN(64, return_sequences=False),
                Dropout(0.5),  # Dropout for regularization
                Dense(64, activation='relu'),
                Dense(len(le.classes_), activation='softmax')  # Output layer
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Model summary
            model.summary()

            # Train the model
            history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(tk.END, f'{algorithm} model trained and saved.\n\n')
    y_pred = model.predict(X_test)
    calculate_metrics(algorithm, y_test, y_pred)

# Calculate metrics
def calculate_metrics(algorithm, testY, predict):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    text.insert(tk.END, f'{algorithm} Accuracy: {a}\n')
    text.insert(tk.END, f'{algorithm} Precision: {p}\n')
    text.insert(tk.END, f'{algorithm} Recall: {r}\n')
    text.insert(tk.END, f'{algorithm} F1 Score: {f}\n\n')
    report = classification_report(testY, predict, target_names=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'])
    text.insert(tk.END, f'Classification Report:\n{report}\n\n')
    conf_matrix = confusion_matrix(testY, predict)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'], yticklabels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'])
    plt.title(f'{algorithm} Confusion Matrix')
    plt.show()

# Prediction
def prediction():
    global model
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(tk.END, f'Loading the Data....\n')
    test = pd.read_csv(filename)
    text.delete('1.0', tk.END)
    test=test.drop(['Unnamed: 0'],axis=1)
    text.insert(tk.END, f'Appling predictions......\n')
    predictions = model.predict(test)
    label_mapping = {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
    relabelled_predictions = np.vectorize(label_mapping.get)(predictions)
    text.delete('1.0', tk.END)
    text.insert(tk.END, 'Predictions:\n')
    for i, p in enumerate(relabelled_predictions):
        text.insert(tk.END, test.iloc[i])
        text.insert(tk.END,f"Row {i}: ********************************************** Predicted as-> {p}")
# GUI Elements
title = Label(main, text=" Dynamic Sentiment Classifier Using Recurrent Neural Networks to Classify Sentiment in Real-Time Across Multiple Languages for Global Markets")
title.grid(column=0, row=0)
font = ('times', 15, 'bold')
title.config(bg='orange', fg='white')
title.config(font=font)
title.config(height=3, width=110)
title.place(x=5, y=5)

text = Text(main, height=25, width=120)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=290, y=100)
text.config(font=('times', 12, 'bold'))

uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.config(bg='blue', fg='Black', width=14)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font)

preprocessButton = Button(main, text="Preprocess Data", command=preprocessing)
preprocessButton.config(bg='blue', fg='Black', width=14)
preprocessButton.place(x=50, y=170)
preprocessButton.config(font=font)

splitButton = Button(main, text="Split Data", command=splitting)
splitButton.config(bg='blue', fg='Black', width=14)
splitButton.place(x=50, y=240)
splitButton.config(font=font)

dtButton = Button(main, text="Decision Tree", command=lambda: train_model('DecisionTreeClassifier'))
dtButton.config(bg='blue', fg='Black', width=14)
dtButton.place(x=50, y=310)
dtButton.config(font=font)

rfButton = Button(main, text="Random Forest", command=lambda: train_model('RandomForestClassifier'))
rfButton.config(bg='blue', fg='Black', width=14)
rfButton.place(x=50, y=380)
rfButton.config(font=font)

mlpButton = Button(main, text="RNN Classifier", command=lambda: train_model('RNNClassifier'))
mlpButton.config(bg='blue', fg='Black', width=14)
mlpButton.place(x=50, y=450)
mlpButton.config(font=font)

predictButton = Button(main, text="Predict", command=prediction)
predictButton.config(bg='blue', fg='Black', width=14)
predictButton.place(x=50, y=520)
predictButton.config(font=font)

main.mainloop()
