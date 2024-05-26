from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = tf.saved_model.load("saved_model/my_model")

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

# def detect_profanity(text):
#     try:
#         print("Text: ", text)
#         preprocessed_text = preprocess_input(text)
#         preprocessed_text = preprocess_input_special_characters(preprocessed_text)
#         print("Preprocessed Text: ", preprocessed_text)
#         # Assume the tokenizer and other preprocessing steps remain the same
#         tokenizer = tf.keras.preprocessing.text.Tokenizer()
#         tokenizer.fit_on_texts([preprocessed_text])
#         print("Tokenizer: ", tokenizer)
#         sequences = tokenizer.texts_to_sequences([preprocessed_text])
#         print("Sequences: ", sequences)
#         padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=128)
#         print("Padded Sequences: ", padded_sequences)
#         # Convert padded_sequences to the correct type
#         padded_sequences = np.array(padded_sequences, dtype=np.int32)

#         # Create the input dictionary as expected by the model
#         input_data = {
#             'input_word_ids': tf.convert_to_tensor(padded_sequences, dtype=tf.int32),
#             'input_mask': tf.convert_to_tensor(np.ones_like(padded_sequences), dtype=tf.int32),
#             'all_segment_id': tf.convert_to_tensor(np.zeros_like(padded_sequences), dtype=tf.int32)
#         }

#         # Use the model to predict
#         predictions = model(input_data, training=False)
        
#         profanity_score = predictions.numpy()[0][0]  # Directly access the tensor value

#         if profanity_score > 0.5:
#             return "Profanity Detected"
#         else:
#             return "No Profanity Detected"
#     except Exception as e:
#         print(e)
#         return str(e)

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

# Load the dataset for Model 2
def load_dataset_model2():
    df_model2 = pd.read_csv('Mau.csv')
    profane_words_model2 = df_model2['words'].tolist()
    return profane_words_model2

# Initialize datasets
profane_words_model2 = load_dataset_model2()
new_df_model1 = load_dataset_model1()

# Initialize TfidfVectorizer for Model 1
tfidf_vectorizer = TfidfVectorizer()
profanity_vectors = tfidf_vectorizer.fit_transform(new_df_model1['Profanity'])

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
        return 'Profanity detected.'
    elif severity_rating == 2:
        return 'Profanity detected.'
    elif severity_rating == 3:
        return 'Profanity detected.'
    else:
        return 'No profanity detected.'

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

def detect_profanity(text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        max_length=128,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    
    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    all_segment_id = inputs['token_type_ids']
    
    # Create a dictionary with the expected keys
    model_inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'all_segment_id': all_segment_id
    }
    
    # Make predictions using the model
    predictions = model(model_inputs, training=False)
    
    profanity_score = (predictions[0] * 10).numpy().tolist() 
    return profanity_score

    # if profanity_score > 0.5:
    #     return "Profanity Detected"
    # else:
    #     return "No Profanity Detected"

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
    model_2_result = detect_profanity_model1(text)

    # Return results as JSON
    return jsonify({
        'manvi': model_2_result,
        'askshat': result
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
