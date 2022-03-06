import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer



def stop_words_check(word_list, stopwords):
    """
    Raison d'être:
        Checks for stopwords in a list of tokens. Note that this will also remove the words with short length
    Args: 
        Word_list a list of word, Stopwords to be checked
    Returns: 
        A list of tokens without stop words
    """
    res_list = []
    for word in word_list: 
        if word not in stopwords and len(word)>3:
            res_list.append(word)
    return res_list

def tokenizer(sentence):
    """
    Raison d'être: 
        Function to use keras' tokenizer. Also remove punctuation, and puts everything in lower case. 
    Args: 
        Sentence to be checked 
    Returns: 
        A list of tokens
    """
    res = tf.keras.preprocessing.text.text_to_word_sequence(
        sentence,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True, 
        split=' '
        )
    return res