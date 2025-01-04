import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from preprocess import preprocess, tokenize, train_val_test_split, create_glove_embeddings, create_embeddings_and_encode
import argparse
import pickle
from save_perplexities import save_perplexities

nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NGramDataset(Dataset):
    def __init__(self, data, embeddings, n=5):
        self.n = n
        self.ngrams = []
        self.labels = []
        self.embeddings = embeddings

        for sentence in data:
            for i in range(len(sentence) - self.n):
                context_indices = sentence[i:i + self.n]
                target_index = sentence[i + self.n]

                context_embeddings = torch.cat([self.embeddings[idx] for idx in context_indices], dim=0).float()
                
                self.ngrams.append(context_embeddings)
                self.labels.append(target_index)

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        return self.ngrams[idx], torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels to Long


class NNLM(nn.Module):
    def __init__(self, embeddings, hidden_dims, n_gram=5, dropout=0.5):
        super(NNLM, self).__init__()

        self.vocab_size = embeddings.shape[0]
        self.embeddings_dim = embeddings.shape[1]

        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.fc1 = nn.Linear((self.embeddings_dim) * n_gram, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], self.vocab_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def test_model(model, eval_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(eval_loader)):
            x, y = x.float().to(device), y.type(torch.LongTensor).to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
        avg_loss = total_loss / len(eval_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience=2):
    model.to(device)
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        train_data = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')

        for x, y in train_data:
            x, y = x.float().to(device), y.type(torch.LongTensor).to(device)  # Ensure Float type

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_train_loss))

        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Train Perplexity: {perplexity:.4f}')

        avg_val_loss, val_perplexity = test_model(model, val_loader, criterion)
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Perplexity: {val_perplexity:.4f}')

        # check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), '2021101072_LM1.pt')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def reload_model():
    print("Reloading model...")
    with open('/Users/ashnadua/Desktop/2021101072_assignment1/data_store/data_store_nnlm.pkl', 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    vocab = data['vocab']
    word_to_idx = data['word_to_idx']
    encoded_train = data['encoded_train']
    encoded_val = data['encoded_val']
    encoded_test = data['encoded_test']

    train_dataset = NGramDataset(encoded_train, embeddings)
    val_dataset = NGramDataset(encoded_val, embeddings)
    test_dataset = NGramDataset(encoded_test, embeddings)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("Data loaded successfully!")

    model = NNLM(embeddings, [300, 300], 5, 0.1)
    model.load_state_dict(torch.load('/Users/ashnadua/Desktop/2021101072_assignment1/models/2021101072_LM1.pt', map_location=device))
    model.eval()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loss, perplexity = test_model(model, val_loader, criterion)
    print(f'\nVal Loss: {loss}')
    print(f'Val Perplexity: {perplexity}\n')

    loss, perplexity = test_model(model, test_loader, criterion)
    print(f'\nTest Loss: {loss}')
    print(f'Test Perplexity: {perplexity}')


def run_model():
    print("Training model...")
    with open('/Users/ashnadua/Desktop/2021101072_assignment1/Auguste_Maquet.txt', 'r') as f:
        corpus = f.read()

    corpus = preprocess(corpus)
    sentences, word_sentences = tokenize(corpus, 6)

    train_sentences, val_sentences, test_sentences = train_val_test_split(word_sentences)

    print("Train size:", len(train_sentences))
    print("Validation size:", len(val_sentences))
    print("Test size:", len(test_sentences))

    glove = create_glove_embeddings('/Users/ashnadua/Desktop/2021101072_assignment1/glove/glove.6B.300d.txt')
    embeddings, encoded_train, encoded_val, encoded_test, word_to_idx, vocab = create_embeddings_and_encode(
        train_sentences, val_sentences, test_sentences, glove
    )

    train_dataset = NGramDataset(encoded_train, embeddings)
    val_dataset = NGramDataset(encoded_val, embeddings)
    test_dataset = NGramDataset(encoded_test, embeddings)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    learning_rate = 0.01
    model = NNLM(embeddings, [300, 300], 5, 0.1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10)

    loss, perplexity = test_model(model, train_loader, criterion)
    print(f'\nTrain Loss: {loss}')
    print(f'Train Perplexity: {perplexity}')

    loss, perplexity = test_model(model, val_loader, criterion)
    print(f'\nVal Loss: {loss}')
    print(f'Val Perplexity: {perplexity}')

    loss, perplexity = test_model(model, test_loader, criterion)
    print(f'\nTest Loss: {loss}')
    print(f'Test Perplexity: {perplexity}')

    with open('/Users/ashnadua/Desktop/2021101072_assignment1/data_store/data_store_nnlm.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'vocab': vocab,
            'word_to_idx': word_to_idx,
            'encoded_train': encoded_train,
            'encoded_val': encoded_val,
            'encoded_test': encoded_test
        }, f)

    print("Data saved successfully!")

    save_perplexities(model, encoded_train, embeddings, criterion, '2021101072_LM1_train_perplexity.txt', vocab, device)
    save_perplexities(model, encoded_val, embeddings, criterion, '2021101072_LM1_val_perplexity.txt', vocab, device)
    save_perplexities(model, encoded_test, embeddings, criterion, '2021101072_LM1_test_perplexity.txt', vocab, device)


def main(args):
    if args.reload:
        reload_model()
    else:
        run_model()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or reload the NNLM model.')
    parser.add_argument('--reload', action='store_true', help='Reload the existing model instead of training a new one.')
    args = parser.parse_args()

    main(args)