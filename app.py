#!/usr/bin/env python
# coding: utf-8

# In[28]:


'''
Prediction 1:
scenario_1: The word test appears in at least in one field
scenario_2: All fields are at most the same
scenario_3: One word field
'''
import pandas as pd
import nltk
from nltk.corpus import words
nltk.download('words')


# In[131]:


# import spam collection messages
names = ['label', 'title']
message_data = pd.read_table("SMSSpamCollection", names=names)


# In[132]:


# Remove rows that contain spam label
message_data = message_data.drop(message_data[message_data.label == "spam"].index)


# In[133]:


# Drop label column
message_data = message_data.drop('label', axis=1)


# In[134]:


message_data = message_data.reset_index(drop=True)
message_data.head(20)


# In[135]:


message_data.shape


# In[136]:


english_words = dict.fromkeys(words.words(), None)

def is_word(word):
    try:
        x = english_words[word]
        return True
    except:
        return False


# In[137]:


indexes = []
print('Starting verification')
for index, message in enumerate(message_data['title']):
    false_word_count = 0
    real_sentence = True
    sentence = message.split()
    for word in sentence:
        if not is_word(word):
            false_word_count += 1
        if false_word_count/len(message) > 0.05:
            indexes.append(index)
            real_sentence = False
            break
print('Done with verification')
print('{} number of indexes will be deleted'.format(len(indexes)))


# In[138]:


real_messages = message_data.drop(message_data.index[indexes])
real_messages.shape


# In[140]:


real_messages.head()


# In[141]:


# Add duplicate message for subtitle and message column
real_messages['sub_title'] = real_messages['title']
real_messages['message'] = real_messages['title']


# In[142]:


real_messages.head()


# In[143]:


real_messages = real_messages.reset_index(drop=True)
real_messages.head()


# In[ ]:




