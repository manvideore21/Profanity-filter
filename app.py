from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from detoxify import Detoxify
from bs4 import BeautifulSoup
from unidecode import unidecode
import tensorflow as tf
from tensorflow import keras

from transformers import BertTokenizer

# Initialize the tokenizer with the model's vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

contractionMap = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

model_path = 'C:/Users/dhira/Desktop/Profanity-filter/OUTPUT_NAME'  # Adjust to your model directory

tfsmlayer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# Define input layers according to the SavedModel's expected input
input_word_ids = keras.Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
input_mask = keras.Input(shape=(128,), dtype=tf.int32, name="input_mask")
segment_ids = keras.Input(shape=(128,), dtype=tf.int32, name="all_segment_id")

# Build the model
outputs = tfsmlayer({'input_word_ids': input_word_ids, 'input_mask': input_mask, 'segment_ids': segment_ids})
model = keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=outputs)


# Load the dataset for Model 1
def load_dataset_model1():
    df = pd.read_csv("Model1.csv")
    data = []
    for value in df['Profanity']:
        for form in ['Canonical Form 1', 'Canonical Form 2', 'Canonical Form 3']:
            if form in df.columns:
                canonical_form_values = df[df['Profanity'] == value][form].dropna().unique()
                for cf_value in canonical_form_values:
                    if ':' in cf_value:
                        profanity_word, severity_rating = cf_value.split(':')
                    else:
                        profanity_word = cf_value.strip()
                        severity_rating = str(df[df['Profanity'] == value]['Severity Rating'].iloc[0])
                    severity_rating = ''.join(filter(str.isdigit, severity_rating))
                    data.append({'Profanity': profanity_word.strip(), 'Severity Rating': int(severity_rating.strip())})
    new_df = pd.DataFrame(data)
    new_df = new_df.drop_duplicates()
    new_df.reset_index(drop=True, inplace=True)
    return new_df

# Initialize datasets
new_df_model1 = load_dataset_model1()

# Initialize TfidfVectorizer for Model 1
tfidf_vectorizer = TfidfVectorizer()
profanity_vectors = tfidf_vectorizer.fit_transform(new_df_model1['Profanity'])

# Preprocessing functions
def removeHTMLTags(text):
    if pd.isnull(text):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    for data in soup(['style', 'script']):
        data.decompose()
    return ' '.join(soup.stripped_strings)

def removeAccentedChars(text):
    return unidecode(text)

def lowercase(text):
    return text.lower()

def removeIPLinkNum(text, ipAddress=True, hyperlink=False, numbers=True):
    if ipAddress:
        text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '', text)
    if hyperlink:
        text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)
    text = re.sub(r'[ ][ ]+', ' ', text)
    return text

def decontract(text):
    for word in contractionMap.keys():
        text = lowercase(text) 
        text = re.sub(word, contractionMap[word], text)
    return text

# Modified detect_profanity function for Model 1
def detect_profanity_model1(sentence):
    # Preprocess input text to replace special characters with alphabets
    sentence = preprocess_input_special_characters(sentence)
    
    sentence_tokens = sentence.split()
    severity_rating = 0
    profanity_count = 0
    max_severity_rating = 0
    for word in sentence_tokens:
        word_vector = tfidf_vectorizer.transform([word])
        similarity_scores = cosine_similarity(word_vector, profanity_vectors)
        max_similarity_score = np.max(similarity_scores)
        if max_similarity_score > 0.8:
            profanity_count += 1
            profanity_word_index = np.argmax(similarity_scores)
            severity_rating += new_df_model1.loc[profanity_word_index, 'Severity Rating']
            max_severity_rating = max(max_severity_rating, new_df_model1.loc[profanity_word_index, 'Severity Rating'])
    if profanity_count > 1:
        severity_rating += max_severity_rating * (profanity_count - 1)
    severity_rating = min(severity_rating, 3)
    if severity_rating == 1:
        return "Mild Profanity - 1"
    elif severity_rating == 2:
        return "Moderate Profanity - 2"
    elif severity_rating == 3:
        return "Severe Profanity - 3"
    else:
        return "No profanity Detected."

# Function to preprocess input text and replace special characters with alphabets
def preprocess_input_special_characters(sentence):
    special_char_mapping = {
        '0': 'o',
        '1': 'i',
        '!': 'i',
        '@': 'a',
        '$': 's',
        '5': 's',
        '7': 't',
        '3': 'e',
        '4': 'a',
        
        # Add more mappings as needed
    }
    for special_char, alphabet in special_char_mapping.items():
        sentence = sentence.replace(special_char, alphabet)
    return sentence

# Load the dataset for Model 2
def load_dataset_model2():
    df_model2 = pd.read_csv('Mau.csv')
    profane_words_model2 = df_model2['words'].tolist()
    return profane_words_model2

# Initialize datasets
profane_words_model2 = load_dataset_model2()

# Preprocess input function
def preprocess_input(text):
    text = removeHTMLTags(text)
    text = removeAccentedChars(text)
    text = lowercase(text)
    text = removeIPLinkNum(text)
    text = decontract(text)  
    return text

# Check for profanity using Model 2
def check_for_profanity(text):
    words = text.split()
    for word in words:
        word = word.lower()
        if word in profane_words_model2:
            return f'Profanity word "{word}" used.'
        if re.search(r'\W', word):
            for profanity_word in profane_words_model2:
                if len(word) == len(profanity_word) and all(a == b or not a.isalnum() for a, b in zip(word, profanity_word)):
                    return f'Warning: Obfuscated profanity word "{word}" used.'
    return "No profanity Detected."

# Check comment toxicity
def check_comment_toxicity(comment):
    model = Detoxify('original')
    results = model.predict(comment)
    if results['toxicity'] > 0.5:
        return "Profanity Detected "
    else:
        return "No Profanity Detected"

@app.route('/')
def home():
    return "Hello, this is the home page."

# @app.route('/profanity-check', methods=['POST'])
# def profanity_check():
#     data = request.get_json()
#     text = data['text']

#     # Model 1
#     model_1_result = detect_profanity_model1(text)

#     # Model 2
#     model_2_result = check_for_profanity(text)

#     # Toxicity Check
#     toxicity_result = check_comment_toxicity(text)

#     # Return results as JSON
#     return jsonify({
#         'model_1_result': model_1_result,
#         'model_2_result': model_2_result,
#         'toxicity_result': toxicity_result
#     })

def process_sentence(sentence, max_seq_length=128):
    """Converts a sentence to input format for BERT."""
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * max_seq_length

    # Padding
    padding_length = max_seq_length - len(input_ids)
    input_ids += [0] * padding_length
    input_mask += [0] * padding_length

    return np.array([input_ids]), np.array([input_mask]), np.array([segment_ids])

# def predict_profanity(text):
#     processed_input = process_input(text)
#     prediction = loaded_model.predict(processed_input)
#     return 'Profane' if prediction[0] > 0.5 else 'Not Profane'


@app.route('/profanity-check', methods=['POST'])
def profanity_check():
    data = request.get_json()
    text = data['text']
    input_word_ids, input_mask, segment_ids = process_sentence(text)
    
    predictions = model.predict([input_word_ids, input_mask, segment_ids])
    
    profanity_score = predictions[0][0]  # Assuming model outputs a single sigmoid output
    result = {
        'profanity_score': float(profanity_score),
        'is_profane': 'Yes' if profanity_score > 0.5 else 'No'
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
