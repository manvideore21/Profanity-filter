from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
from tensorflow.keras.layers import TFSMLayer

# Load the new model
model = TFSMLayer('C:/Users/91805/Desktop/Profanity-filter/OUTPUT_NAME', call_endpoint='serving_default')



# # Load model from C:\Users\91805\Desktop\Profanity-filter\OUTPUT_NAME
# model_path = "C:/Users/91805/Desktop/Profanity-filter/OUTPUT_NAME"
# model = tf.keras.layers.load_model(model_path, call_endpoint = "serving_default")
# model = tf.keras.models.load_model('model_path')

# print(model.summary())

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

# Preprocess input function
def preprocess_input(text):
    text = removeHTMLTags(text)
    text = removeAccentedChars(text)
    text = lowercase(text)
    text = removeIPLinkNum(text)
    text = decontract(text)  
    return text

# Modify detect_profanity to use the new layer format
def detect_profanity(text):
    try:
        print("hit 1")

        preprocessed_text = preprocess_input(text)
        preprocessed_text = preprocess_input_special_characters(preprocessed_text)
        print("hit 2")
        
        # Assume the tokenizer and other preprocessing steps remain the same
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts([preprocessed_text])
        sequences = tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=128)  # Adjust maxlen to 128 as per model requirement
        print("hit 3")
        # Make predictions using the model
        predictions = model(padded_sequences)
        print("hit 4")
        profanity_score = np.max(predictions.numpy())  # Make sure to convert the tensor output to numpy if necessary
        print("hit 5")
        
        if profanity_score > 0.5:
            return "Profanity Detected"
        else:
            return "No Profanity Detected"
    except Exception as e:
        print(e)

@app.route('/')
def home():
    return "Hello, this is the home page."

@app.route('/profanity-check', methods=['POST'])
def profanity_check():
    data = request.get_json()
    text = data['text']

    print(text)

    # Profanity detection
    result = detect_profanity(text)

    # Return results as JSON
    return jsonify({
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
