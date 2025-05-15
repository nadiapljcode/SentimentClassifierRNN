# SentimentClassifierRNN
This project implements a Recurrent Neural Network (RNN)-based sentiment analysis system that dynamically classifies user opinions across various markets and languages. It focuses on extracting meaningful sentiments from multilingual text data

# Dynamic Sentiment Classification Using RNN
* A Tkinter-based GUI application for sentiment analysis across global markets, leveraging machine learning algorithms and Recurrent Neural Networks (RNNs). This tool classifies multilingual textual data (like tweets or reviews) into sentiment categories (1–5 stars) and supports real-time predictions using pre-trained models.

---

# 📌 Features
* Upload and preprocess datasets with stopword removal, stemming, and TF-IDF
* Multilingual support via Label Encoding
* Class balancing using SMOTE
* Train/test split and model training
* Multiple ML classifiers: Decision Tree, Random Forest, Extra Trees
* Deep Learning: Recurrent Neural Network (RNN) with Keras
* Model evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
* Predict unseen data with GUI interface
* Model persistence using joblib
---

# ⚙️ Technologies Used

- Languages & Libraries: Python, NLTK, Scikit-learn, Imbalanced-learn, TensorFlow/Keras
- GUI: Tkinter
- Visualization: Matplotlib, Seaborn
- NLP: Tokenization, Stemming, Lemmatization, TF-IDF
- Balancing: SMOTE
- Models:
    DecisionTreeClassifier,
    RandomForestClassifier,
    Recurrent Neural Network (SimpleRNN)
---

# 📂 Project Structure


```plaintext
sentiment-analysis-project/
│
├── .gitignore
├── README.md
├── requirements.txt
├── run.bat
├── main.py
│
├── model/
│   ├── DecisionTreeClassifier.pkl
│   ├── ExtraTreesClassifier.pkl
│   ├── RandomForestClassifier.pkl
│   ├── RNNClassifier.pkl
│   └── rnnClassifier.h5
│
├── data/
│   ├── data.csv
│   └── tests.csv
│
├── notebooks/
│   ├── Untitled.ipynb
│   └── Untitled1.ipynb
│
├── .venv/
└── catboost_info/
```

