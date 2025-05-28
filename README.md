# 📰 Fake News Detection Using Machine Learning

This project aims to classify whether a news article is **real** or **fake** using natural language processing and machine learning techniques. It evaluates and compares the performance of several supervised models including:

- Logistic Regression : Linear model for binary classification.
- Decision Tree Classifier : Simple tree-based classifier.
- Random Forest Classifier : Ensemble of decision trees, reduces overfitting.
- Gradient Boosting Classifier : Builds models sequentially to reduce error.

---

## 📊 Dataset

The dataset contains:
- `title`: News article title.
- `text`: News article content.
- `subject`: Subject of news.
- `date`: Date.
- `class`: Label (0 = Fake, 1 = Real).

Example sources:
- [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 🧹 Text Preprocessing

Text data is cleaned using a custom function:

```python
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
```

---
## 🔢 Feature Extraction

Used TF-IDF Vectorizer from sklearn to convert cleaned text into numeric features:

```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorization = TfidfVectorizer()
  xv_train = vectorization.fit_transform(z_train)
  xv_test = vectorization.transform(x_test)
```

---




