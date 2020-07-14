########################
# NLTK install on Jupyter
########################

pip install nltk
import nltk
nltk.download()
--> select "all" option

########################
# Code execution 
########################

import collections
import nltk
import os 
import random
import codecs

# funcion to read from disk
def load_files(directory):
    result =[]
    for fname in os.listdir(directory):
        with open(directory + fname ,'r' , encoding='utf-8',
                 errors='ignore' ) as f:
            result.append(f.read())
    return result

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words('english')) 
stop_words

def preprocss_sentence(sentence):

# tokenize words
    processed_tokens = nltk.word_tokenize(sentence)
# covert to lower case
    processed_tokens = [w.lower() for w in processed_tokens]
    
    word_counts = collections.Counter(processed_tokens)
    uncommon_words = word_counts.most_common()[:-10:-1]
    
    processed_tokens = [w for w in processed_tokens if w not in stop_words]
    processed_tokens = [w for w in processed_tokens if w not in uncommon_words]
    
    # lemmatization ( remove same words having different forms)
    lemmatize = WordNetLemmatizer()
    processed_tokens = [lemmatize.lemmatize(w) for w in processed_tokens]
    return processed_tokens
    
    # read  emails from folder
positive_example = load_files('/home/jovyan/demo/ham/')
negative_example = load_files('/home/jovyan/demo/spam/')

 positive_example = [preprocss_sentence(email) for email in positive_example]
 negative_example = [preprocss_sentence(email) for email in negative_example]

# mark spams as 0 and hams as 1
positive_example =[(email,1) for email in positive_example]
negative_example =[(email,0) for email in negative_example]

all_examples = positive_example + negative_example

random.shuffle(all_examples)

def feature_extraction(tokens):
    return dict(collections.Counter(tokens))

featurized = [(feature_extraction(corpus), label) for corpus, label in all_examples]

def test_train_split(dataset, train_size=0.8):
    num_training_example =  int(len(dataset) * train_size)
    return dataset[:num_training_example],dataset[num_training_example:]
    
    training_set, test_set = test_train_split (featurized , train_size =0.7)
    
    model = nltk.classify.NaiveBayesClassifiertrain(training_set)
    training_error = nltk.classify.accuracy(model,training_set)
    testing_error = nltk.classify.accuracy(model,test_set)
    