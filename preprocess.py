import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import torch
import numpy as np

def preprocess(data):
    data = re.sub(r'\n|\s+', ' ', data) #newline and multiple spaces -> single space
    data = re.sub(r'[’‘]', '\'', data) #apostrophes
    data = re.sub(r'[“”`\' ]|[–—-]', ' ', data) #quotes and dashes
    data = re.sub(r'(?<!\w)([.!?])(?!\w)', r' \1 ', data) #dont remove punctuation
    data = re.sub(r'[™•]', ' ', data) #remove other unwanted symbols
    return data.strip() #strip extra spaces


def tokenize(data, min_length_sentences):
    sentences = sent_tokenize(data)
    sentences = [sentence for sentence in sentences if len(sentence.split()) >= min_length_sentences]
    print("Length of sentences after filtering:", len(sentences))

    words_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        
        words = [word.lower() for word in words if word.lower() not in ['.', ',', '!', '?', ';', ':']]
        words = ['<s>'] + words + ['</s>']
        words_sentences.append(words)
    
    return sentences, words_sentences


def train_val_test_split(sentences, train_ratio=0.7, val_ratio=0.2, seed=None, num_shuffles=1):
    if seed is not None:
        random.seed(seed)
    
    for _ in range(num_shuffles):
        random.shuffle(sentences)
    
    total_sentences = len(sentences)
    
    train_size = int(total_sentences * train_ratio)
    val_size = int(total_sentences * val_ratio)
    test_size = total_sentences - train_size - val_size  # Remaining for test
    
    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:train_size + val_size]
    test_sentences = sentences[train_size + val_size:]
    
    return train_sentences, val_sentences, test_sentences


def create_glove_embeddings(glove_path):
    glove = {}
    embedding_dim = 0

    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]])
            glove[word] = vector
            embedding_dim = len(values[1:])

    glove['<UNK>'] = torch.mean(torch.stack(list(glove.values())), dim=0)
    glove['<s>'] = torch.rand(embedding_dim)
    glove['</s>'] = torch.rand(embedding_dim)

    return glove


def create_embeddings_and_encode(train_sentences, val_sentences, test_sentences, glove):
    embedding_dim = len(list(glove.values())[0])
    vocab = set()

    # create vocab from train
    vocab.update(['<s>', '</s>', '<UNK>', '<PAD>'])

    for sentence in train_sentences:
        for word in sentence:
            if word in glove:
                vocab.add(word)
            else:
                sentence[sentence.index(word)] = '<UNK>'
                
    embeddings = np.zeros((len(vocab), embedding_dim))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    for word in vocab:
        if word in glove:
            embeddings[word_to_idx[word]] = glove[word]
        else:
            embeddings[word_to_idx[word]] = np.random.rand(embedding_dim)  # Random for unknown words

    def encode_sentences(sentences, word_to_idx):
        encoded_sentences = []
        for sentence in sentences:
            encoded_sentence = [word_to_idx[word] if word in word_to_idx else word_to_idx['<UNK>'] for word in sentence]
            encoded_sentences.append(encoded_sentence)
        return encoded_sentences

    encoded_train_sentences = encode_sentences(train_sentences, word_to_idx)
    encoded_val_sentences = encode_sentences(val_sentences, word_to_idx)
    encoded_test_sentences = encode_sentences(test_sentences, word_to_idx)

    return torch.FloatTensor(embeddings), encoded_train_sentences, encoded_val_sentences, encoded_test_sentences, word_to_idx, list(vocab)