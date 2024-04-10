# # import pandas as pd
# # df = pd.read_csv("Profanity Dataset.csv")
# # df["Severity Rating"].describe()
# # unique_counts = df.apply(lambda x: x.nunique())
# # total_unique_words = unique_counts.sum()
# # import pandas as pd
# # data = []
# # for value in df['Profanity']:
# #     if 'Canonical Form 1' in df.columns:
# #         canonical_form_1_values = df[df['Profanity'] == value]['Canonical Form 1'].dropna().unique()
# #         for cf1_value in canonical_form_1_values:
# #             if ':' in cf1_value:
# #                 profanity_word, severity_rating = cf1_value.split(':')
# #             else:
# #                 profanity_word = cf1_value.strip()
# #                 severity_rating = str(df[df['Profanity'] == value]['Severity Rating'].iloc[0])
# #             severity_rating = ''.join(filter(str.isdigit, severity_rating))
# #             data.append({'Profanity': profanity_word.strip(), 'Severity Rating': int(severity_rating.strip())})
# #     if 'Canonical Form 2' in df.columns:
# #         canonical_form_2_values = df[df['Profanity'] == value]['Canonical Form 2'].dropna().unique()
# #         for cf2_value in canonical_form_2_values:
# #             if ':' in cf2_value:
# #                 profanity_word, severity_rating = cf2_value.split(':')
# #             else:
# #                 profanity_word = cf2_value.strip()
# #                 severity_rating = str(df[df['Profanity'] == value]['Severity Rating'].iloc[0])
# #             severity_rating = ''.join(filter(str.isdigit, severity_rating))
# #             data.append({'Profanity': profanity_word.strip(), 'Severity Rating': int(severity_rating.strip())})
# #     if 'Canonical Form 3' in df.columns:
# #         canonical_form_3_values = df[df['Profanity'] == value]['Canonical Form 3'].dropna().unique()
# #         for cf3_value in canonical_form_3_values:
# #             if ':' in cf3_value:
# #                 profanity_word, severity_rating = cf3_value.split(':')
# #             else:
# #                 profanity_word = cf3_value.strip()
# #                 severity_rating = str(df[df['Profanity'] == value]['Severity Rating'].iloc[0])
# #             severity_rating = ''.join(filter(str.isdigit, severity_rating))
# #             data.append({'Profanity': profanity_word.strip(), 'Severity Rating': int(severity_rating.strip())})
# # new_df = pd.DataFrame(data)
# # new_df = new_df.drop_duplicates()
# # new_df.reset_index(drop=True, inplace=True)
# # len(new_df)
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np
# # new_df.head()
# # profanity_words = []
# # for column in new_df.columns:
# #     if column != 'Severity Rating':
# #         profanity_words.extend(new_df[column].dropna().unique())
# # profanity_words = list(set(profanity_words))
# # print(profanity_words)
# # len(profanity_words)
# # if "cock" in profanity_words:
# #     print("Found in the list.")
# # else:
# #     print("Did not find in the list.")
# # tfidf_vectorizer = TfidfVectorizer()
# # profanity_vectors = tfidf_vectorizer.fit_transform(profanity_words)
# # def detect_profanity(sentence):
# #     sentence_tokens = sentence.split()
# #     severity_rating = 0
# #     profanity_count = 0
# #     max_severity_rating = 0
# #     for word in sentence_tokens:
# #         word_vector = tfidf_vectorizer.transform([word])
# #         similarity_scores = cosine_similarity(word_vector, profanity_vectors)
# #         max_similarity_score = np.max(similarity_scores)
# #         if max_similarity_score > 0.8:
# #             profanity_count += 1
# #             profanity_word_index = np.argmax(similarity_scores)
# #             severity_rating += new_df.loc[profanity_word_index, 'Severity Rating']
# #             max_severity_rating = max(max_severity_rating, new_df.loc[profanity_word_index, 'Severity Rating'])
# #     if profanity_count > 1:
# #         severity_rating += max_severity_rating * (profanity_count - 1)
# #     severity_rating = min(severity_rating, 3)
# #     return severity_rating
# # input_sentence = "Fuck you bitch"
# # severity_rating = detect_profanity(input_sentence)
# # if severity_rating > 0:
# #     print("Severity Rating:", severity_rating)
# # else:
# #     print("The sentence does not contain profanity.")
# # input_sentence = "black man"
# # word_vector = tfidf_vectorizer.transform([input_sentence])
# # similarity_scores = cosine_similarity(word_vector, profanity_vectors)
# # max_similarity_score = np.max(similarity_scores)
# # print(max_similarity_score)
# # sentence = "hello fucking bitch"
# # word_list = sentence.split()
# # print(word_list)
# # for word in word_list:
# #     severity_rating = detect_profanity(word)
# #     rating = str(severity_rating)
# #     if severity_rating > 0:
# #         print(f"Word '{word}' has profanity. Severity Rating:", rating)
# #     else:
# #         print(f"Word '{word}' does not contain profanity.")





# #!/usr/bin/env python
# # coding: utf-8

# # In[15]:


# from unidecode import unidecode
# import re
# import pandas as pd


# # In[16]:


# df = pd.read_csv("Model1.csv")


# # In[17]:


# df.head()


# # In[18]:


# df["Severity Rating"].describe()


# # In[19]:


# unique_counts = df.apply(lambda x: x.nunique())
# print(unique_counts)
# total_unique_words = unique_counts.sum()
# print("Total number of unique words in all columns:", total_unique_words)


# # In[20]:


# df.describe()


# # In[21]:


# df.tail()


# # In[22]:


# import pandas as pd


# unique_values = pd.unique(df[["Profanity", "Canonical Form 1", "Canonical Form 2", "Canonical Form 3"]].values.ravel('K'))

# new_data = []
# for value in unique_values:
#     temp_df = df[df.isin([value]).any(axis=1)]
#     if not temp_df.empty:
#         severity_rating = temp_df.iloc[0]["Severity Rating"]
#         if pd.notna(value):  
#             new_data.append([value, severity_rating])

# new_df = pd.DataFrame(new_data, columns=["Profanity", "Severity Rating"])

# print(new_df)
# len(new_df)


# # In[23]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# # In[24]:


# new_df.head()


# # In[25]:


# profanity_words = []
# for column in new_df.columns:
#     if column != 'Severity Rating':
#         profanity_words.extend(new_df[column].dropna().unique())
# profanity_words = list(set(profanity_words))
# print(profanity_words)


# # In[26]:


# len(profanity_words)


# # In[27]:


# if "oral" in profanity_words:
#     print("Found in the list.")
# else:
#     print("Did not find in the list.")


# # In[28]:


# tfidf_vectorizer = TfidfVectorizer()
# profanity_vectors = tfidf_vectorizer.fit_transform(profanity_words)


# # In[29]:


# def detect_profanity(sentence):
#     sentence_tokens = sentence.split()
#     severity_rating = 0
#     profanity_count = 0
#     max_severity_rating = 0
#     for word in sentence_tokens:
#         if word.lower() in ['tch', 'shi']:
#             continue
#         word_vector = tfidf_vectorizer.transform([word])
#         similarity_scores = cosine_similarity(word_vector, profanity_vectors)
#         max_similarity_score = np.max(similarity_scores)
#         if max_similarity_score > 0.80: 
#             profanity_count += 1
#             profanity_word_index = np.argmax(similarity_scores)
#             severity_rating += new_df.loc[profanity_word_index, 'Severity Rating']
#             max_severity_rating = max(max_severity_rating, new_df.loc[profanity_word_index, 'Severity Rating'])
    
#     if profanity_count > 1:
#         severity_rating = 3
    
#     severity_rating = min(severity_rating, 3)
    
#     return severity_rating


# # In[30]:


# input_sentence = "sh"
# severity_rating = detect_profanity(input_sentence)
# if severity_rating > 0:
#     print("Severity Rating:", severity_rating)
# else:
#     print("The sentence does not contain profanity.")


# # In[31]:


# def removeHTMLTags(text):
#     '''
#     Function to remove the HTML Tags from a given text.
#     '''
#     if pd.isnull(text):
#         return ""
#     soup = BeautifulSoup(text, 'html.parser')
#     for data in soup(['style', 'script']):
#         data.decompose()
#     return ' '.join(soup.stripped_strings)


# # In[32]:


# def removeAccentedChars(text):
#     '''
#     Function to remove the accented characters from a given text.
#     '''
#     return unidecode(text)


# # In[33]:


# def lowercase(text):
#     '''
#     Function to convert a given text to its lowercase.
#     '''
    
#     return text.lower()


# # In[34]:


# def removeIPLinkNum(text, ipAddress=True, hyperlink=False, numbers=True):
#     '''
#     Function to remove IP Address and Number from the given text.
#     '''
   
#     if ipAddress == True:
        
#         text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '', text)
    
#     if hyperlink == True:
        
#         text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)
    
#     # if numbers == True:
        
#     #     text = re.sub(r'[0-9]', '', text)
    
#     text = re.sub(r'[ ][ ]+', ' ', text)
    
#     return text


# # In[35]:


# contractionMap = {
#     'ain\'t': 'is not',
#     'aren\'t': 'are not',
#     'can\'t': 'cannot',
#     'can\'t\'ve': 'cannot have',
#     '\'cause': 'because',
#     'could\'ve': 'could have',
#     'couldn\'t': 'could not',
#     'couldn\'t\'ve': 'could not have',
#     'didn\'t': 'did not',
#     'doesn\'t': 'does not',
#     'don\'t': 'do not',
#     'hadn\'t': 'had not',
#     'hadn\'t\'ve': 'had not have',
#     'hasn\'t': 'has not',
#     'haven\'t': 'have not',
#     'he\'d': 'he would',
#     'he\'d\'ve': 'he would have',
#     'he\'ll': 'he will',
#     'he\'ll\'ve': 'he he will have',
#     'he\'s': 'he is',
#     'how\'d': 'how did',
#     'how\'d\'y': 'how do you',
#     'how\'ll': 'how will',
#     'how\'s': 'how is',
#     'I\'d': 'I would',
#     'I\'d\'ve': 'I would have',
#     'I\'ll': 'I will',
#     'I\'ll\'ve': 'I will have',
#     'I\'m': 'I am',
#     'I\'ve': 'I have',
#     'i\'d': 'i would',
#     'i\'d\'ve': 'i would have',
#     'i\'ll': 'i will',
#     'i\'ll\'ve': 'i will have',
#     'i\'m': 'i am',
#     'i\'ve': 'i have',
#     'isn\'t': 'is not',
#     'it\'d': 'it would',
#     'it\'d\'ve': 'it would have',
#     'it\'ll': 'it will',
#     'it\'ll\'ve': 'it will have',
#     'it\'s': 'it is',
#     'let\'s': 'let us',
#     'ma\'am': 'madam',
#     'mayn\'t': 'may not',
#     'might\'ve': 'might have',
#     'mightn\'t': 'might not',
#     'mightn\'t\'ve': 'might not have',
#     'must\'ve': 'must have',
#     'mustn\'t': 'must not',
#     'mustn\'t\'ve': 'must not have',
#     'needn\'t': 'need not',
#     'needn\'t\'ve': 'need not have',
#     'o\'clock': 'of the clock',
#     'oughtn\'t': 'ought not',
#     'oughtn\'t\'ve': 'ought not have',
#     'shan\'t': 'shall not',
#     'sha\'n\'t': 'shall not',
#     'shan\'t\'ve': 'shall not have',
#     'she\'d': 'she would',
#     'she\'d\'ve': 'she would have',
#     'she\'ll': 'she will',
#     'she\'ll\'ve': 'she will have',
#     'she\'s': 'she is',
#     'should\'ve': 'should have',
#     'shouldn\'t': 'should not',
#     'shouldn\'t\'ve': 'should not have',
#     'so\'ve': 'so have',
#     'so\'s': 'so as',
#     'that\'d': 'that would',
#     'that\'d\'ve': 'that would have',
#     'that\'s': 'that is',
#     'there\'d': 'there would',
#     'there\'d\'ve': 'there would have',
#     'there\'s': 'there is',
#     'they\'d': 'they would',
#     'they\'d\'ve': 'they would have',
#     'they\'ll': 'they will',
#     'they\'ll\'ve': 'they will have',
#     'they\'re': 'they are',
#     'they\'ve': 'they have',
#     'to\'ve': 'to have',
#     'wasn\'t': 'was not',
#     'we\'d': 'we would',
#     'we\'d\'ve': 'we would have',
#     'we\'ll': 'we will',
#     'we\'ll\'ve': 'we will have',
#     'we\'re': 'we are',
#     'we\'ve': 'we have',
#     'weren\'t': 'were not',
#     'what\'ll': 'what will',
#     'what\'ll\'ve': 'what will have',
#     'what\'re': 'what are',
#     'what\'s': 'what is',
#     'what\'ve': 'what have',
#     'when\'s': 'when is',
#     'when\'ve': 'when have',
#     'where\'d': 'where did',
#     'where\'s': 'where is',
#     'where\'ve': 'where have',
#     'who\'ll': 'who will',
#     'who\'ll\'ve': 'who will have',
#     'who\'s': 'who is',
#     'who\'ve': 'who have',
#     'why\'s': 'why is',
#     'why\'ve': 'why have',
#     'will\'ve': 'will have',
#     'won\'t': 'will not',
#     'won\'t\'ve': 'will not have',
#     'would\'ve': 'would have',
#     'wouldn\'t': 'would not',
#     'wouldn\'t\'ve': 'would not have',
#     'y\'all': 'you all',
#     'y\'all\'d': 'you all would',
#     'y\'all\'d\'ve': 'you all would have',
#     'y\'all\'re': 'you all are',
#     'y\'all\'ve': 'you all have',
#     'you\'d': 'you would',
#     'you\'d\'ve': 'you would have',
#     'you\'ll': 'you will',
#     'you\'ll\'ve': 'you will have',
#     'you\'re': 'you are',
#     'you\'ve': 'you have'
# }


# # In[36]:


# def decontract(text):
#     '''
#     Function to decontract a given text.
#     '''
#     for word in contractionMap.keys():

#         text = lowercase(text) 
#         text = re.sub(word, contractionMap[word], text)
        
#     return text


# # In[37]:


# from nltk.corpus import words
# from bs4 import BeautifulSoup
# word_list = words.words()
# def preprocess_word_list(word_list):
#     processed_word_list = []
#     for word in word_list:
#         word = removeHTMLTags(word)
#         word = removeAccentedChars(word)
#         word = lowercase(word)
#         word = removeIPLinkNum(word)
#         processed_word_list.append(word)
#     return processed_word_list

# word_list = preprocess_word_list(word_list)


# # In[38]:


# # for word in word_list:
# #     severity_rating = detect_profanity(word)
# #     rating = str(severity_rating)
# #     if severity_rating > 0:
# #         print(f"Word '{word}' has profanity. Severity Rating:", rating)


# # In[39]:


# from Levenshtein import distance

# def detect_profanity_Lev(sentence, profanity_list):
#     sentence_tokens = sentence.split()
#     severity_rating = 0
#     profanity_count = 0
#     max_severity_rating = 0
#     for word in sentence_tokens:
#         if word.lower() in ['tch', 'shi']:
#             continue
#         for profane_word in profanity_list:
#             if distance(word.lower(), profane_word.lower()) <= 0.8:  # adjust threshold 
#                 profanity_count += 1
#                 profanity_word_index = profanity_list.index(profane_word)
#                 severity_rating += new_df.loc[profanity_word_index, 'Severity Rating']
#                 max_severity_rating = max(max_severity_rating, new_df.loc[profanity_word_index, 'Severity Rating'])
    
#     if profanity_count > 1:
#         severity_rating = 3
    
#     return severity_rating


# # In[40]:


# # for word in word_list:
# #     severity_rating = detect_profanity_Lev(word, profanity_words)
# #     rating = str(severity_rating)
# #     if severity_rating > 0:
# #         print(f"Word '{word}' has profanity. Severity Rating:", rating)


# # In[43]:


# def preprocess_input(text):
#     text = removeHTMLTags(text)
#     text = removeAccentedChars(text)
#     text = lowercase(text)
#     text = removeIPLinkNum(text)
#     text = decontract(text)  
    
#     return text

# input_sentence = "co-coon"
# preprocessed_sentence = preprocess_input(input_sentence)

# severity_rating_lev = detect_profanity_Lev(preprocessed_sentence, profanity_words)
# severity_rating_tfidf = detect_profanity(preprocessed_sentence)

# if severity_rating_lev > 0:
#     print("Levenshtein-based Profanity Detected. Severity Rating:", severity_rating_lev)
# else:
#     print("No Levenshtein-based Profanity Detected.")

# if severity_rating_tfidf > 0:
#     print("TF-IDF Similarity-based Profanity Detected. Severity Rating:", severity_rating_tfidf)
# else:
#     print("No TF-IDF Similarity-based Profanity Detected.")


# # In[34]:


# # visually similar characters in model 


# # In[ ]:










#!/usr/bin/env python
# coding: utf-8






from unidecode import unidecode
import re
import pandas as pd

df = pd.read_csv("Model1.csv")

df.head()

df["Severity Rating"].describe()

unique_counts = df.apply(lambda x: x.nunique())
print(unique_counts)
total_unique_words = unique_counts.sum()
print("Total number of unique words in all columns:", total_unique_words)

df.describe()

df.tail()

import pandas as pd

unique_values = pd.unique(df[["Profanity", "Canonical Form 1", "Canonical Form 2", "Canonical Form 3"]].values.ravel('K'))

new_data = []
for value in unique_values:
    temp_df = df[df.isin([value]).any(axis=1)]
    if not temp_df.empty:
        severity_rating = temp_df.iloc[0]["Severity Rating"]
        if pd.notna(value):  
            new_data.append([value, severity_rating])

new_df = pd.DataFrame(new_data, columns=["Profanity", "Severity Rating"])

print(new_df)
len(new_df)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

new_df.head()

profanity_words = []
for column in new_df.columns:
    if column != 'Severity Rating':
        profanity_words.extend(new_df[column].dropna().unique())
profanity_words = list(set(profanity_words))
print(profanity_words)

len(profanity_words)

if "oral" in profanity_words:
    print("Found in the list.")
else:
    print("Did not find in the list.")

tfidf_vectorizer = TfidfVectorizer()
profanity_vectors = tfidf_vectorizer.fit_transform(profanity_words)

def detect_profanity(sentence):
    sentence_tokens = sentence.split()
    severity_rating = 0
    profanity_count = 0
    max_severity_rating = 0
    for word in sentence_tokens:
        if word.lower() in ['tch', 'shi']:
            continue
        word_vector = tfidf_vectorizer.transform([word])
        similarity_scores = cosine_similarity(word_vector, profanity_vectors)
        max_similarity_score = np.max(similarity_scores)
        if max_similarity_score > 0.80: 
            profanity_count += 1
            profanity_word_index = np.argmax(similarity_scores)
            severity_rating += new_df.loc[profanity_word_index, 'Severity Rating']
            max_severity_rating = max(max_severity_rating, new_df.loc[profanity_word_index, 'Severity Rating'])
    
    if profanity_count > 1:
        severity_rating = 3
    
    severity_rating = min(severity_rating, 3)
    
    return severity_rating

input_sentence = "sh"
severity_rating = detect_profanity(input_sentence)
if severity_rating > 0:
    print("Severity Rating:", severity_rating)
else:
    print("The sentence does not contain profanity.")

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
   
    if ipAddress == True:
        
        text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '', text)
    
    if hyperlink == True:
        
        text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)
    
    # if numbers == True:
        
    #     text = re.sub(r'[0-9]', '', text)
    
    text = re.sub(r'[ ][ ]+', ' ', text)
    
    return text

contractionMap = {
    'ain\'t': 'is not',
    'aren\'t': 'are not',
    'can\'t': 'cannot',
    'can\'t\'ve': 'cannot have',
    '\'cause': 'because',
    'could\'ve': 'could have',
    'couldn\'t': 'could not',
    'couldn\'t\'ve': 'could not have',
    'didn\'t': 'did not',
    'doesn\'t': 'does not',
    'don\'t': 'do not',
    'hadn\'t': 'had not',
    'hadn\'t\'ve': 'had not have',
    'hasn\'t': 'has not',
    'haven\'t': 'have not',
    'he\'d': 'he would',
    'he\'d\'ve': 'he would have',
    'he\'ll': 'he will',
    'he\'ll\'ve': 'he he will have',
    'he\'s': 'he is',
    'how\'d': 'how did',
    'how\'d\'y': 'how do you',
    'how\'ll': 'how will',
    'how\'s': 'how is',
    'I\'d': 'I would',
    'I\'d\'ve': 'I would have',
    'I\'ll': 'I will',
    'I\'ll\'ve': 'I will have',
    'I\'m': 'I am',
    'I\'ve': 'I have',
    'i\'d': 'i would',
    'i\'d\'ve': 'i would have',
    'i\'ll': 'i will',
    'i\'ll\'ve': 'i will have',
    'i\'m': 'i am',
    'i\'ve': 'i have',
    'isn\'t': 'is not',
    'it\'d': 'it would',
    'it\'d\'ve': 'it would have',
    'it\'ll': 'it will',
    'it\'ll\'ve': 'it will have',
    'it\'s': 'it is',
    'let\'s': 'let us',
    'ma\'am': 'madam',
    'mayn\'t': 'may not',
    'might\'ve': 'might have',
    'mightn\'t': 'might not',
    'mightn\'t\'ve': 'might not have',
    'must\'ve': 'must have',
    'mustn\'t': 'must not',
    'mustn\'t\'ve': 'must not have',
    'needn\'t': 'need not',
    'needn\'t\'ve': 'need not have',
    'o\'clock': 'of the clock',
    'oughtn\'t': 'ought not',
    'oughtn\'t\'ve': 'ought not have',
    'shan\'t': 'shall not',
    'sha\'n\'t': 'shall not',
    'shan\'t\'ve': 'shall not have',
    'she\'d': 'she would',
    'she\'d\'ve': 'she would have',
    'she\'ll': 'she will',
    'she\'ll\'ve': 'she will have',
    'she\'s': 'she is',
    'should\'ve': 'should have',
    'shouldn\'t': 'should not',
    'shouldn\'t\'ve': 'should not have',
    'so\'ve': 'so have',
    'so\'s': 'so as',
    'that\'d': 'that would',
    'that\'d\'ve': 'that would have',
    'that\'s': 'that is',
    'there\'d': 'there would',
    'there\'d\'ve': 'there would have',
    'there\'s': 'there is',
    'they\'d': 'they would',
    'they\'d\'ve': 'they would have',
    'they\'ll': 'they will',
    'they\'ll\'ve': 'they will have',
    'they\'re': 'they are',
    'they\'ve': 'they have',
    'to\'ve': 'to have',
    'wasn\'t': 'was not',
    'we\'d': 'we would',
    'we\'d\'ve': 'we would have',
    'we\'ll': 'we will',
    'we\'ll\'ve': 'we will have',
    'we\'re': 'we are',
    'we\'ve': 'we have',
    'weren\'t': 'were not',
    'what\'ll': 'what will',
    'what\'ll\'ve': 'what will have',
    'what\'re': 'what are',
    'what\'s': 'what is',
    'what\'ve': 'what have',
    'when\'s': 'when is',
    'when\'ve': 'when have',
    'where\'d': 'where did',
    'where\'s': 'where is',
    'where\'ve': 'where have',
    'who\'ll': 'who will',
    'who\'ll\'ve': 'who will have',
    'who\'s': 'who is',
    'who\'ve': 'who have',
    'why\'s': 'why is',
    'why\'ve': 'why have',
    'will\'ve': 'will have',
    'won\'t': 'will not',
    'won\'t\'ve': 'will not have',
    'would\'ve': 'would have',
    'wouldn\'t': 'would not',
    'wouldn\'t\'ve': 'would not have',
    'y\'all': 'you all',
    'y\'all\'d': 'you all would',
    'y\'all\'d\'ve': 'you all would have',
    'y\'all\'re': 'you all are',
    'y\'all\'ve': 'you all have',
    'you\'d': 'you would',
    'you\'d\'ve': 'you would have',
    'you\'ll': 'you will',
    'you\'ll\'ve': 'you will have',
    'you\'re': 'you are',
    'you\'ve': 'you have'
}

def decontract(text):
    for word in contractionMap.keys():

        text = lowercase(text) 
        text = re.sub(word, contractionMap[word], text)
        
    return text

from nltk.corpus import words
from bs4 import BeautifulSoup
word_list = words.words()
def preprocess_word_list(word_list):
    processed_word_list = []
    for word in word_list:
        word = removeHTMLTags(word)
        word = removeAccentedChars(word)
        word = lowercase(word)
        word = removeIPLinkNum(word)
        processed_word_list.append(word)
    return processed_word_list

word_list = preprocess_word_list(word_list)

from Levenshtein import distance

def detect_profanity_Lev(sentence, profanity_list):
    sentence_tokens = sentence.split()
    severity_rating = 0
    profanity_count = 0
    max_severity_rating = 0
    for word in sentence_tokens:
        if word.lower() in ['tch', 'shi']:
            continue
        for profane_word in profanity_list:
            if distance(word.lower(), profane_word.lower()) <= 0.8:  # adjust threshold 
                profanity_count += 1
                profanity_word_index = profanity_list.index(profane_word)
                severity_rating += new_df.loc[profanity_word_index, 'Severity Rating']
                max_severity_rating = max(max_severity_rating, new_df.loc[profanity_word_index, 'Severity Rating'])
    
    if profanity_count > 1:
        severity_rating = 3
    
    return severity_rating

for word in word_list:
    severity_rating = detect_profanity_Lev(word, profanity_words)
    rating = str(severity_rating)
    if severity_rating > 0:
        print(f"Word '{word}' has profanity. Severity Rating:", rating)

def preprocess_input(text):
    text = removeHTMLTags(text)
    text = removeAccentedChars(text)
    text = lowercase(text)
    text = removeIPLinkNum(text)
    text = decontract(text)  
    
    return text

input_sentence = "co-coon"
preprocessed_sentence = preprocess_input(input_sentence)

severity_rating_lev = detect_profanity_Lev(preprocessed_sentence, profanity_words)
severity_rating_tfidf = detect_profanity(preprocessed_sentence)

if severity_rating_lev > 0:
    print("Levenshtein-based Profanity Detected. Severity Rating:", severity_rating_lev)
else:
    print("No Levenshtein-based Profanity Detected.")

if severity_rating_tfidf > 0:
    print("TF-IDF Similarity-based Profanity Detected. Severity Rating:", severity_rating_tfidf)
else:
    print("No TF-IDF Similarity-based Profanity Detected.")
