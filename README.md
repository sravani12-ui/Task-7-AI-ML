# Fake News Detection Model

This project uses machine learning techniques to classify news articles as either *real* or *fake* based on text data. Leveraging Natural Language Processing (NLP) techniques and a machine learning classifier, this model helps in identifying misinformation from reliable news sources.

## Overview
The model uses the **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization method to convert text data into numerical features and a **Linear Support Vector Classifier (LinearSVC)** to classify news articles. With an accuracy of approximately 99%, this model provides a solid baseline for detecting fake news.

## Dataset
The dataset consists of two classes:
- **Real**: Legitimate news articles
- **Fake**: Misinformation or unreliable news articles
- **Text**: The news article text.
- **Label**: The classification label (Real/Fake).

### Data Distribution
A bar chart visualizing class distribution is provided to show the balance between real and fake news articles in the dataset.

![DD](Image/class_distribution.png)

## Dependencies
To run this project, ensure you have the following libraries installed:
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`

Install dependencies using:
```bash
pip install -r requirements.txt
```
## Model Training and Evaluation

1. TF-IDF Vectorization: Converts the article text into a numerical format suitable for machine learning.
2. Model Training: A LinearSVC model is trained on the vectorized data.
3. Evaluation: The model is evaluated using accuracy and classification metrics, along with a confusion matrix for visual representation.

# Key Code Sections:
**Import required librariess**

```bash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
### Data Loading:
```bash
# Load dataset
df = pd.read_csv('/content/fake_and_real_news.csv')
```
**Dataset Size Check**

 df.shape, shows the dimensions of the dataset (number of rows and columns). It helps us understand the dataset's size.
 ```bash
df.shape
 ```
output: (9900, 2) #9900 data and 2 column 

**First five row:**

![df](Image/dataset.png)

## Splitting Data for Training and Testing
This code splits the data into training and testing sets, with 80% for training and 20% for testing. The split is done randomly, but the random_state=0 ensures consistent results each time.

```bash
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
```
Display a pie chart showing the percentage of data used for training and testing, helping you visualize the split.

```bash
import matplotlib.pyplot as plt

# Plot the proportion of training and testing data
labels = ['Train', 'Test']
sizes = [len(X_train), len(X_test)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Train-Test Split Proportion')
plt.savefig('/content/train_test_split.png')
plt.show()
```
![pie](Image/train_test_split.png)

## Importing SVM and Classifier

imports two classification models:
- LinearSVC is used for linear support vector classification.
- SGDClassifier is used for classification with stochastic gradient descent.

```bash
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
```

## TF-IDF Transformation:
TF-IDF Vectorizer that converts text into numerical features. It limits the features to the top 5000 most important words based on TF-IDF scores. It then transforms the test data using the same fitted vectorizer.

```bash
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer

# Fit the vectorizer on the training data and transform both X_train and X_test
X_train_tfidf = vectorizer.fit_transform(X_train['Text'])
X_test_tfidf = vectorizer.transform(X_test['Text'])
```

## LinearSVC Classifier

initializes a LinearSVC model and trains it using the transformed training data (X_train_tfidf) and the labels (Y_train). The random state ensures consistent results.

```bash
clf = LinearSVC(random_state=0)
clf.fit(X_train_tfidf, Y_train)
```
**Making Prediction with classifier:**

```bash
Y_pred = clf.predict(X_test_tfidf)
```

## Evaluating Model Accuracy
```bash
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)
print("Accuracy:", accuracy)
```
**Output:** Accuracy: 0.9984848484848485

##Classification Report

This classification report shows that the model did a perfect job at telling "Fake" from "Real." Hereâ€™s what each part means:

- Precision: For both "Fake" and "Real" categories, precision is 1.00, meaning every time the model labeled something as "Fake" or "Real," it was correct.
- Recall: Recall is also 1.00 for both categories, so the model found every "Fake" and every "Real" instance without missing any.
- F1-Score: This combines precision and recall, and since both are perfect, the F1-score is 1.00 as well, showing excellent balance.
- Support: There were 1,017 "Fake" cases and 963 "Real" cases, giving a total of 1,980 samples.
 ```bash
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the classification report
report = classification_report(Y_test, Y_pred, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Plot the classification report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f', cbar=True)
plt.title('Classification Report - Lung Cancer Prediction')
plt.savefig('Lr report.png')
plt.show()
```
![cr](Image/Lr%20report.png) 

## Confusion Matrix:
The confusion matrix provides a look at how well the model correctly identified "Fake" and "Real" cases:

- True Positives (1015): The model correctly classified 1,015 "Fake" cases as "Fake."
- True Negatives (962): It also correctly identified 962 "Real" cases as "Real."
- False Positives (2): There were 2 instances where the model incorrectly labeled "Real" cases as "Fake."
- False Negatives (1): Only 1 "Fake" case was mislabeled as "Real.".

```bash
# Generate confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(Y_test, Y_pred)
# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.show()
```
![CM](Image/confusion_matrix.png)

## How to Use
1. **Clone this repository:**
```bash
git@github.com:TaskiyaMridha/Fake_news_Classification.git
```
2. **Navigate to the project directory:**
```bash
cd fake_news_detection
```
3. **Install dependancies:**
```bash
pip install -r requirements.txt
```
4. **Run the notebook or script to load the dataset, train the model, and evaluate its performance.**

## Future Improvements

* Experiment with additional models like Random Forest or Naive Bayes.
* Use more complex NLP techniques, like word embeddings, to improve classification accuracy.
* Explore ensemble models for potential accuracy improvements.
