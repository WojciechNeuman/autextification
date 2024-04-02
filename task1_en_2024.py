#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from py3langid.langid import LanguageIdentifier, MODEL_FILE
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

import warnings

# Suppress FutureWarning related to is_sparse
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


train = pd.read_csv('data/competition_2024/subtask1.csv').drop(columns='Unnamed: 0')

train.head()


# In[3]:


train.loc[0, 'text']


# In[4]:


identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE)
identifier.set_languages(['en', 'es', 'pt', 'gl', 'eu', 'ca'])

# English (en)
# Spanish (es)
# Portuguese (pt)
# Galician (gl)
# Basque (eu)
# Catalan (ca)

identifier.classify(train.loc[0, 'text'])


# In[5]:


def detect_language(text):
    return identifier.classify(text)[0]

# Create the new column
train['detected_language'] = train['text'].apply(detect_language)

train


# In[6]:


train['label'].value_counts()


# In[7]:


train['detected_language'].value_counts()


# In[8]:


train_en = train[train['detected_language'] == 'en']
train_es = train[train['detected_language'] == 'es']
train_pt = train[train['detected_language'] == 'pt']
train_gl = train[train['detected_language'] == 'gl']
train_eu = train[train['detected_language'] == 'eu']
train_ca = train[train['detected_language'] == 'ca']


# In[9]:


import string
from nltk.corpus import stopwords

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    STOPWORDS = set(stopwords.words('english'))
    # STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

# import nltk 
# nltk.download('stopwords') # # -> do this in case of 'Resource stopwords not found.'


def preprocessing(df):
    # processing text
    df.loc[:, 'clean_text'] = df['text'].apply(text_process)

    # calculate text length
    df.loc[:, 'text_len'] = df.loc[:, 'clean_text'].apply(len)
    
    # crop unnecessary columns
    df = df.drop(columns=['text', 'detected_language'])

    # map labels
    df['label'] = df['label'].map({'human': 0, 'generated': 1})

    return df


# In[10]:


train_en_preprocessed = preprocessing(train_en)


# In[11]:


train_en_preprocessed


# In[12]:


plt.figure(figsize=(12, 8))

train_en_preprocessed[train_en_preprocessed.label == 0].text_len.plot(bins=35, kind='hist', color='blue', 
                                       label='Human texts', alpha=0.5)
train_en_preprocessed[train_en_preprocessed.label == 1].text_len.plot(bins=35, kind='hist', color='red', 
                                       label='Generated texts', alpha=0.5)
plt.legend()
plt.xlabel("Message Length")


# In[13]:


def plot_common_words(ham_words, type):
    common_words = [word[0] for word in ham_words.most_common(20)]
    word_counts = [word[1] for word in ham_words.most_common(20)]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(common_words, word_counts, color='skyblue')
    plt.title(f'Top 20 Most Common Words in {type} Labeled Data')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# In[14]:


words = train_en_preprocessed[train_en_preprocessed.label == 0]['clean_text'].apply(lambda x: [word.lower() for word in x.split()])
ham_words_human = Counter()

for msg in words:
    ham_words_human.update(msg)
    
print(ham_words_human.most_common(50))


# In[15]:


words = train_en_preprocessed[train_en_preprocessed.label == 1]['clean_text'].apply(lambda x: [word.lower() for word in x.split()])
ham_words_computer = Counter()

for msg in words:
    ham_words_computer.update(msg)
    
print(ham_words_computer.most_common(50))


# In[16]:


for counter in [(ham_words_human, 'Human'), (ham_words_computer, 'Computer')]:
    plot_common_words(counter[0], counter[1])


# In[17]:


print(f"Average Human text length: {train_en_preprocessed[train_en_preprocessed.label == 0]['text_len'].mean()}")
print(f"Average Computer text length: {train_en_preprocessed[train_en_preprocessed.label == 1]['text_len'].mean()}")


# In[18]:


# train_en_preprocessed.loc[:, train_en_preprocessed.columns != 'label'
X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(train_en_preprocessed.loc[:, 'clean_text'], 
                                                                train_en_preprocessed['label'], test_size=0.20, random_state=42)

print(X_train_en.shape)
print(X_test_en.shape)
print(y_train_en.shape)
print(y_test_en.shape)


# In[19]:


X_train_en


# In[20]:


y_train_en


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer

# instantiate the vectorizer
vect = CountVectorizer()
vect.fit(X_train_en)

# learn training data vocabulary, then use it to create a document-term matrix
X_train1_dtm = vect.transform(X_train_en)

# equivalently: combine fit and transform into a single step
X_train1_dtm = vect.fit_transform(X_train_en)


# examine the document-term matrix
print(type(X_train1_dtm), X_train1_dtm.shape)

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test1_dtm = vect.transform(X_test_en)
print(type(X_test1_dtm), X_test1_dtm.shape)


# In[22]:


# from sklearn.feature_extraction.text import TfidfTransformer

# tfidf_transformer = TfidfTransformer()
# tfidf_transformer.fit(X_train1_dtm)
# tfidf_transformer.transform(X_train1_dtm)


# In[23]:


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[24]:


# train the model using X_train1_dtm (timing it with an IPython "magic command")
get_ipython().run_line_magic('time', 'nb.fit(X_train1_dtm, y_train_en)')


# In[25]:


from sklearn import metrics

# make class predictions for X_test_dtm
y_test_en_pred_mb = nb.predict(X_test1_dtm)

# calculate accuracy of class predictions
print("=======Accuracy Score===========")
print(metrics.accuracy_score(y_test_en, y_test_en_pred_mb))

# print the confusion matrix
print("=======Confision Matrix===========")
metrics.confusion_matrix(y_test_en, y_test_en_pred_mb)


# In[26]:


from sklearn.metrics import f1_score
print(f"F1-score for Naive Bayes model: {f1_score(y_test_en, y_test_en_pred_mb, average='macro')}")


# In[27]:


# import an instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')

# train the model using X_train_dtm
get_ipython().run_line_magic('time', 'logreg.fit(X_train1_dtm, y_train_en)')


# In[28]:


# make class predictions for X_test_dtm
y_test_en_pred_lr = logreg.predict(X_test1_dtm)

# calculate predicted probabilities for X_test_dtm (well calibrated)
y_test_en_prob_lr = logreg.predict_proba(X_test1_dtm)[:, 1]
y_test_en_prob_lr


# In[29]:


# calculate accuracy of class predictions
print("=======Accuracy Score===========")
print(metrics.accuracy_score(y_test_en, y_test_en_pred_lr))

# print the confusion matrix
print("=======Confision Matrix===========")
print(metrics.confusion_matrix(y_test_en, y_test_en_pred_lr))

# calculate AUC
print("=======ROC AUC Score===========")
print(metrics.roc_auc_score(y_test_en, y_test_en_pred_lr))


# In[30]:


from sklearn.metrics import f1_score
print(f"F1-score for Logistic Regression model: {f1_score(y_test_en, y_test_en_pred_lr, average='macro')}")

