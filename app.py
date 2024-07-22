from flask import Flask, render_template, request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Step 1: Load and preprocess the data
zip_file_path = r"C:\Users\91846\Desktop\spam\spam.zip"  # Update with your actual file path
csv_file_name = "spam.csv"  # Just the file name, not the full path

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall("data_folder")

data = pd.read_csv(f"data_folder/{csv_file_name}", encoding='ISO-8859-1')

data.drop_duplicates(inplace=True)
data.dropna(subset=['v2'], inplace=True)
data.reset_index(drop=True, inplace=True)
data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    email_text_transformed = tfidf_vectorizer.transform([email_text])
    prediction = classifier.predict(email_text_transformed)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
