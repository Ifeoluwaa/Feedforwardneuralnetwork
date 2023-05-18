import string
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words("english"))


#Reading the file
def read_file(texts):
    with open(texts, 'r') as fo:
        file = fo.read()
    list_of_lines = file.splitlines()
    corpus = []
    for sentence in list_of_lines:
        if sentence != "":
            corpus.append(sentence)
    return corpus
    
    
#Cleaning the data; changing to lower case
def clean_data(sentence):
    sentence = sentence.lower()
    return sentence 


##word tokenization
def tokenize(sentence):
    listofsentence = word_tokenize(sentence)
    return listofsentence


#tokenizing the entire corpus
def tokenizecorpus(listofsentence):
    tokenize_corpus = [word_tokenize(sentence) for sentence in listofsentence]
    return tokenize_corpus


#frequency of words in the above tokenization
def word_frequency(tokenize_corpus):
    words_term_frequency_train = {}
    for line in tokenize_corpus:
        for word in line:
            if word not in words_term_frequency_train:
                words_term_frequency_train[word] = 1
            else:
                words_term_frequency_train[word] = words_term_frequency_train.get(word, 0) + 1
    return words_term_frequency_train


#Vocabulary
def create_vocab(tokenize_corpus, unk=False):
    vocab = []
    for sentence in tokenize_corpus:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)

    return vocab


def create_context_and_labels(data, n_grams):
    data_size = len(data)
    context = n_grams - 1
    dataset = []
    for i in range(context, data_size - context):
        counter = context
        context_words = []
        while counter >= 1:
            context_words.append(data[i - counter])
            counter -= 1
        dataset.append((context_words, data[i]))
    return dataset


def idx_to_word(idx, word_to_index_map):
    for key, value in word_to_index_map.items():
        if value == idx:
            return key
        
        
        
#Mapping of words to numbers
def word_to_index_map(vocabulary):
    word_to_ix = {}
    for i, word in enumerate(vocabulary):
        word_to_ix[word] = i
    return word_to_ix


#spliting the dataset to train and test 
def train_test_split(corpus, train_size, shuffle=False):
    if shuffle:
        random.shuffle(corpus)
    train_size = int(train_size * len(corpus))
    train_arr = corpus[:train_size]
    test_arr = corpus[train_size + 1:]
    return train_arr, test_arr

    

# getting the mapped words
def get_id_word(word, word_id_map):
    return word_id_map.get(word, 0) 
