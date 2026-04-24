
# Fake News Classifier & Deployment

## Overview

This project focuses on detecting fake news articles using Machine Learning techniques.
The text content of news articles is processed using Natural Language Processing (NLP) methods such as text cleaning, stopword removal, and TF-IDF vectorization.

Two classification models were trained and compared to determine which model performs best in distinguishing between **Fake** and **Real** news articles. The best-performing model was then deployed using a **Gradio web application**, allowing users to test the model by entering news text.

## Dataset

* **Features used:** Text (news article content)
* **Target:** label (Fake / Real)
* **Total samples:** 9900

## Model Comparison

| Model                   | Accuracy | Precision | Recall | F1-Score |
| ----------------------- | -------- | --------- | ------ | -------- |
| Multinomial Naive Bayes | 0.969    | 0.97      | 0.97   | 0.97     |
| Logistic Regression     | 0.996    | 1.00      | 0.99   | 1.00     |

## Final Model

**Model:** Logistic Regression
**Accuracy:** 0.996

### Why this model?

Logistic Regression was selected as the final model because it achieved the **highest accuracy, precision, recall, and F1-score** compared to Multinomial Naive Bayes.
It also showed better generalization on the test dataset and produced fewer misclassifications in the confusion matrix.

## Web Application

The trained model is deployed using **Gradio**, which provides a simple web interface where users can input news text and instantly see whether the news is predicted as **Fake** or **Real**.

### Screenshot

![Gradio Interface](screenshots/gradio_interface .png)

---

## Installation

Clone the repository:

```
git clone [https://github.com/kanogeeeepust-coder/fake---news---classifier.git]
cd fake-news-classifier
```

Install required libraries:

```
pip install -r requirements.txt
```

## Usage

Run the web application:

```
python app.py
```

Then open the local Gradio link in your browser to test the model.

## Project Structure

```
fake-news-classifier/
│
├── data/
│
├── notebooks/
│   ├── 1_eda.ipynb
│   └── 2_training.ipynb
│
├── models/
│
├── screenshots/
│   └── gradio_interface.png
│
├── app.py
├── README.md
└── requirements.txt
```

## Technologies Used

* Python
* Pandas
* NLTK
* Matplotlib
* Seaborn
* Scikit-learn
* Gradio
