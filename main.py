# =========================================
# NLP Assignment
# Complete NLP Processing on PDF
# =========================================

# Install Required Libraries First:
# py -m pip install PyPDF2 nltk scikit-learn pandas plotly

# =========================================
# Import Libraries
# =========================================

import PyPDF2
import re
import string
import pandas as pd
import nltk
import plotly.express as px

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================
# Download NLTK Data
# =========================================

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# =========================================
# Q1(a): PDF Reading and Text Extraction
# =========================================


print("Q1(a): PDF Reading")

# Open PDF File
pdf_file = open("NLPArticle.pdf", "rb")

# Read PDF
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Total Pages
total_pages = len(pdf_reader.pages)

print("\nTotal Pages in PDF:", total_pages)

# Extract Text from All Pages
all_text = ""

for page in pdf_reader.pages:
    extracted_text = page.extract_text()
    
    if extracted_text:
        all_text += extracted_text

# Show Sample Extracted Text
print("\nSample Extracted Text:\n")
print(all_text[:1000])

# =================
# Q1(b): Text Preprocessing
# =========================================

print("Q1(b): Text Preprocessing")

# Convert Text to Lowercase
text = all_text.lower()

print("\nLowercase Conversion Completed")

# -----------------------------------------
# Remove Numbers Using Regex
# Regex Pattern: \d+
# -----------------------------------------

text = re.sub(r'\d+', '', text)

print("\nNumbers Removed Using Regex")
print("Regex Pattern Used: \\d+")

# -----------------------------------------
# Remove Special Symbols Using Regex
# Regex Pattern: [^a-zA-Z\s]
# -----------------------------------------

text = re.sub(r'[^a-zA-Z\s]', ' ', text)

print("\nSpecial Symbols Removed Using Regex")
print("Regex Pattern Used: [^a-zA-Z\\s]")

# -----------------------------------------
# Remove Extra Spaces Using Regex
# Regex Pattern: \s+
# -----------------------------------------

text = re.sub(r'\s+', ' ', text).strip()

print("\nExtra Spaces Removed Using Regex")
print("Regex Pattern Used: \\s+")

# -----------------------------------------
# Remove Punctuation
# -----------------------------------------

text = text.translate(
    str.maketrans('', '', string.punctuation)
)

print("\nPunctuation Removed")

# =========================================
# Tokenization
# =========================================

tokens = word_tokenize(text)

print("\nTotal Tokens:", len(tokens))

print("\nSample Tokens:")
print(tokens[:20])

# =========================================
# Stop Word Removal
# =========================================

stop_words = set(stopwords.words('english'))

valid_words = []
stop_word_count = 0

for word in tokens:
    
    if word not in stop_words:
        valid_words.append(word)
    else:
        stop_word_count += 1

print("\nTotal Stop Words Found:", stop_word_count)

print("Valid Words After Stop Word Removal:",
      len(valid_words))

print("\nSample Valid Words:")
print(valid_words[:20])

# =========================================
# Stemming
# =========================================

stemmer = PorterStemmer()

stemmed_words = []

for word in valid_words:
    stemmed_words.append(
        stemmer.stem(word)
    )

print("\nSample Stemmed Words:")
print(stemmed_words[:20])

# =========================================
# Lemmatization
# =========================================

lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for word in valid_words:
    lemmatized_words.append(
        lemmatizer.lemmatize(word)
    )

print("\nSample Lemmatized Words:")
print(lemmatized_words[:20])

# =========================================
# Q1(c): One Hot Encoding
# =========================================

print("Q1(c): One Hot Encoding")

# Take Sample Words
sample_words = list(set(valid_words[:20]))

# Create DataFrame
onehot_df = pd.DataFrame(
    sample_words,
    columns=['Words']
)

# One Hot Encoder
encoder = OneHotEncoder(
    sparse_output=False
)

# Fit and Transform
encoded = encoder.fit_transform(
    onehot_df[['Words']]
)

# Convert to DataFrame
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(['Words'])
)

print("\nOne Hot Encoding Output:\n")

print(encoded_df)

# =========================================
# TF-IDF Feature Extraction
# =========================================


print("TF-IDF Feature Extraction")

# Join Cleaned Words
cleaned_text = " ".join(valid_words)

# Create Vectorizer
vectorizer = TfidfVectorizer()

# Fit and Transform
tfidf_matrix = vectorizer.fit_transform(
    [cleaned_text]
)

# Get Feature Names
feature_names = vectorizer.get_feature_names_out()

# Get TF-IDF Scores
tfidf_scores = tfidf_matrix.toarray()[0]

# Create DataFrame
tfidf_df = pd.DataFrame({
    'Word': feature_names,
    'TF-IDF Score': tfidf_scores
})

# Sort by Highest Score
tfidf_df = tfidf_df.sort_values(
    by='TF-IDF Score',
    ascending=False
)

print("\nTop 20 TF-IDF Features:\n")

print(tfidf_df.head(20))

# =========================================
# Word Frequency
# =========================================

word_frequency = Counter(valid_words)

freq_df = pd.DataFrame({
    'Word': list(word_frequency.keys()),
    'Frequency': list(word_frequency.values())
})

freq_df = freq_df.sort_values(
    by='Frequency',
    ascending=False
)

top_freq_words = freq_df.head(30)

# =========================================
# Q1(d): Plotly Graphs
# =========================================

print("Q1(d): Plotly Visualizations")

# =========================================
# Graph 1: Scatter Plot
# =========================================

top_tfidf = tfidf_df.head(30)

fig1 = px.scatter(
    top_tfidf,
    x='Word',
    y='TF-IDF Score',
    title='TF-IDF Scatter Plot'
)

fig1.update_layout(
    xaxis_title='Words',
    yaxis_title='TF-IDF Scores'
)

fig1.show()

# =========================================
# Graph 2: Bar Chart
# =========================================

fig2 = px.bar(
    top_tfidf,
    x='Word',
    y='TF-IDF Score',
    title='Top TF-IDF Words Bar Chart'
)

fig2.update_layout(
    xaxis_title='Words',
    yaxis_title='TF-IDF Scores'
)

fig2.show()

# =========================================
# Graph 3: Line Chart
# =========================================

fig3 = px.line(
    top_freq_words,
    x='Word',
    y='Frequency',
    title='Word Frequency Line Chart'
)

fig3.update_layout(
    xaxis_title='Words',
    yaxis_title='Frequency'
)

fig3.show()

# =========================================
# Graph 4: Pie Chart
# =========================================

fig4 = px.pie(
    top_freq_words,
    names='Word',
    values='Frequency',
    title='Top Word Frequency Pie Chart'
)

fig4.show()

# =========================================
# End Message
# =========================================

print("Assignment Completed Successfully")