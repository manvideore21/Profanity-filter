from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})
CORS(app)

# model = tf.saved_model.load("saved_model/my_model")

# Load the dataset for Model 2
def load_dataset_model2():
    df_model2 = pd.read_csv('Mau.csv')
    profane_words_model2 = df_model2['words'].tolist()
    return profane_words_model2

# Initialize datasets
profane_words_model2 = load_dataset_model2()

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

# def detect_profanity(text):
#     # Tokenize the input text
#     inputs = tokenizer.encode_plus(
#         text,
#         max_length=128,
#         truncation=True,
#         padding='max_length',
#         add_special_tokens=True,
#         return_tensors='tf'
#     )
    
#     input_word_ids = inputs['input_ids']
#     input_mask = inputs['attention_mask']
#     all_segment_id = inputs['token_type_ids']
    
#     # Create a dictionary with the expected keys
#     model_inputs = {
#         'input_word_ids': input_word_ids,
#         'input_mask': input_mask,
#         'all_segment_id': all_segment_id
#     }
    
#     # Make predictions using the model
#     predictions = model(model_inputs, training=False)
    
#     profanity_score = (predictions[0] * 10).numpy().tolist() 
#     return profanity_score

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
    # result = detect_profanity(text)
    model_2_result = check_for_profanity(text)

    # Return results as JSON
    return jsonify({
        'manvi': model_2_result,
        # 'askshat': result
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
