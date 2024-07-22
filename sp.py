import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Define the path to the ZIP file
zip_file_path = r"C:\Users\91846\Desktop\spam\spam.zip"  # Update with your actual file path

# Define the name of the CSV file inside the ZIP archive
csv_file_name = "spam.csv"  # Just the file name, not the full path

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall("data_folder")  # Specify the folder where files will be extracted

# Load the CSV data with 'ISO-8859-1' encoding
data = pd.read_csv(f"data_folder/{csv_file_name}", encoding='ISO-8859-1')

# Step 2: Data Preprocessing
# Customize this section based on your specific data preprocessing needs
data.drop_duplicates(inplace=True)
data.dropna(subset=['v2'], inplace=True)  # Assuming 'v2' is the email text column
data.reset_index(drop=True, inplace=True)

# Rename columns for clarity if needed
data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

# Step 3: Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text'])  # 'text' is the email text column
y = data['label']  # 'label' is the column indicating spam or non-spam

# Step 4: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection and Step 6: Model Training
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


