
import pandas as pd
import re
# Load the dataset
df = pd.read_csv('Mau.csv')
profane_words = df['words'].tolist()
def contains_profanity(text):
    for word in profane_words:
        if word in text.lower(): # Convert text to lowercase to make the check case-insensitive
            return True
    return False

def contains_profanity(text):
    for word in profane_words:
        # Ensure both text and word are strings before performing the check
        if isinstance(word, str) and isinstance(text, str) and word in text.lower():
            return True
    return False

profane_words = df['words'].tolist()


def check_for_profanity(sentence):
    words = sentence.split()

    for word in words:
        word = word.lower()

        # Check for exact match
        if word in profane_words:
            return f'Profanity word "{word}" used.'

        # Check for obfuscated profanity
        if re.search(r'\W', word):  # Check for any non-alphanumeric character
            for profanity_word in profane_words:

                if len(word) == len(profanity_word) and all(a == b or not a.isalnum() for a, b in zip(word, profanity_word)):
                    return f'Obfuscated profanity word "{word}" used.'
                

    return "No profanity detected."



# Take input from the user
user_input = input("Please enter a string: ")

# Check if the input contains profanity
result = check_for_profanity(user_input)
print(result)
