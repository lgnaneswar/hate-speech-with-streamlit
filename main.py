# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
import pytesseract
from PIL import Image
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Additional resource for WordNet Lemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import requests

# Initialize NLTK lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopword = set(stopwords.words('english'))

# Function to clean text data with lemmatization
def clean(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove square brackets and their contents
    text = re.sub(r'\[.*?\]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)

    # Remove punctuation
    text = re.sub(rf'[{string.punctuation}]', '', text)

    # Remove newline characters
    text = re.sub(r'\n', '', text)

    # Remove words containing digits
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove stopwords and apply lemmatization
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stopword]
    text = " ".join(text)
    
    return text

# Set up pytesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract text from image using pytesseract
def extract_text(image_file):
    # Open the image file
    image = Image.open(image_file)
    
    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(image)
    
    return text

# Load your data from CSV file
data = pd.read_csv("twitter.csv")

# Map numerical labels to descriptive categories
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})

# Keep only relevant columns
data = data[["tweet", "labels"]]

# Apply the cleaning function to the "tweet" column
data["tweet"] = data["tweet"].apply(clean)

# Split the data into features (x) and labels (y)
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Initialize CountVectorizer for feature extraction
cv = CountVectorizer()

# Transform text data into a bag-of-words representation
X = cv.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)
