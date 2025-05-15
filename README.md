# Dynamic Sentiment Classification Using RNN
* A Tkinter-based GUI application for sentiment analysis across global markets, leveraging machine learning algorithms and Recurrent Neural Networks (RNNs). This tool classifies multilingual textual data (like tweets or reviews) into sentiment categories (1–5 stars) and supports real-time predictions using pre-trained models.

  
## 📌 Features
* Upload and preprocess datasets with stopword removal, stemming, and TF-IDF
* Multilingual support via Label Encoding
* Class balancing using SMOTE
* Train/test split and model training
* Multiple ML classifiers: Decision Tree, Random Forest, Extra Trees
* Deep Learning: Recurrent Neural Network (RNN) with Keras
* Model evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
* Predict unseen data with GUI interface
* Model persistence using joblib  


## ⚙️ Technologies Used

- Languages & Libraries: Python, NLTK, Scikit-learn, Imbalanced-learn, TensorFlow/Keras
- GUI: Tkinter
- Visualization: Matplotlib, Seaborn
- NLP: Tokenization, Stemming, Lemmatization, TF-IDF
- Balancing: SMOTE
- Models:
    DecisionTreeClassifier,
    RandomForestClassifier,
    Recurrent Neural Network (SimpleRNN)


## 📂 Project Structure

```
sentiment-analysis-project/
│
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

## 🛠️ Architecture 
![image](https://github.com/user-attachments/assets/8ea40846-2c7e-496b-ad01-c390fd72dc2c)
   
## Performance Analysis
- Here we have compared the proposed algorithm (RNN) with baseline algorithms to evaluate performance using various metrics and confusion matrices.

### Confusion Matrix - Decision Tree Classifier
<img width="401" alt="image" src="https://github.com/user-attachments/assets/43c922ac-0471-41d3-a12d-3652d07e8d93" />

### Confusion Matrix - Random Forest Classifier
<img width="375" alt="image" src="https://github.com/user-attachments/assets/0e3afc4b-f72d-4219-ba79-044ff869ad98" />

### Confusion Matrix - Recurrent Neural Network
<img width="418" alt="image" src="https://github.com/user-attachments/assets/62cd2601-b977-4ca8-a697-02a72a5a2e3b" />


- From the confusion matrix, we observe that RNN provides better sentiment classification performance in most categories

## 💻 Screenshots

## 🚀 Getting Started
1. Clone the repo
2. Install requirements:  
   `pip install -r requirements.txt`
3. Run:  
   `run.bat`
