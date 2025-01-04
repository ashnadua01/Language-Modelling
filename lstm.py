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
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from preprocess import preprocess, tokenize, train_val_test_split, create_glove_embeddings, create_embeddings_and_encode
import argparse
import pickle
from save_perplexities import save_perplexities_lstm

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        input_sentence = torch.tensor(sentence[:-1], dtype=torch.long)
        target = torch.tensor(sentence[1:], dtype=torch.long)
        return input_sentence, target
    

def collate_fn(batch, pad_idx):
    input_sentences, targets = zip(*batch)
    input_sentences = pad_sequence(input_sentences, batch_first=True, padding_value=pad_idx)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    return input_sentences, targets


class LSTM(nn.Module):
    def __init__(self, embeddings, hidden_dim, dropout, num_layers=1):
        super(LSTM, self).__init__() 
        # freeze embeddings
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, input_seq, hidden=None):
        input_seq = self.embeddings(input_seq)
        
        if hidden is None:
            lstm_out, hidden = self.lstm(input_seq)
        else:
            lstm_out, hidden = self.lstm(input_seq, hidden)
            
        return self.fc1(self.dropout(lstm_out)), hidden


def test_model(model, val_loader, criterion, pad_idx):
    model.eval()
    total_loss = 0
    hidden = None

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            if hidden is not None and batch_size != hidden[0].size(1):
                hidden = None

            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, output.shape[2]), y.view(-1))
            total_loss += loss.item()

            if hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())

    avg_val_loss = total_loss / len(val_loader)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    return avg_val_loss, val_perplexity


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience=2, pad_idx=0):
    model.to(device)
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        hidden = None

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            if hidden is not None and batch_size != hidden[0].size(1):
                hidden = None

            optimizer.zero_grad()
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, output.shape[2]), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())

        avg_train_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_train_loss))

        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Train Perplexity: {perplexity:.4f}')

        avg_val_loss, val_perplexity = test_model(model, val_loader, criterion, pad_idx)

        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Perplexity: {val_perplexity:.4f}')
        
        # check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), '2021101072_LM2.pt')
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
    with open('/Users/ashnadua/Desktop/2021101072_assignment1/data_store/data_store_lstm.pkl', 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    vocab = data['vocab']
    word_to_idx = data['word_to_idx']
    encoded_train = data['encoded_train']
    encoded_val = data['encoded_val']
    encoded_test = data['encoded_test']

    train_dataset = LSTMDataset(encoded_train)
    val_dataset = LSTMDataset(encoded_val)
    test_dataset = LSTMDataset(encoded_test)

    pad_idx = word_to_idx['<PAD>']

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)

    print("Data loaded successfully!")

    model = LSTM(embeddings, 300, 0.5, 2)
    model.load_state_dict(torch.load('/Users/ashnadua/Desktop/2021101072_assignment1/models/2021101072_LM2.pt', map_location=device))

    model.eval()
    model.to(device)
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    loss, perplexity = test_model(model, val_loader, criterion, pad_idx)
    print(f'\nVal Loss: {loss}')
    print(f'Val Perplexity: {perplexity}\n')                         

    loss, perplexity = test_model(model, test_loader, criterion, pad_idx)
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

    pad_idx = word_to_idx['<PAD>']

    train_dataset = LSTMDataset(encoded_train)
    val_dataset = LSTMDataset(encoded_val)
    test_dataset = LSTMDataset(encoded_test)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    learning_rate = 0.001
    num_epochs = 10
    patience = 1
    model = LSTM(embeddings, 300, 0.5, 2)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    model = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience)

    loss, perplexity = test_model(model, train_loader, criterion, pad_idx)
    print(f'\nTrain Loss: {loss}')
    print(f'Train Perplexity: {perplexity}\n')

    loss, perplexity = test_model(model, val_loader, criterion, pad_idx)
    print(f'\nVal Loss: {loss}')
    print(f'Val Perplexity: {perplexity}\n')

    loss, perplexity = test_model(model, test_loader, criterion, pad_idx)
    print(f'\nTest Loss: {loss}')
    print(f'Test Perplexity: {perplexity}\n')

    save_perplexities_lstm(model, encoded_train, criterion, '2021101072_LM2_train_perplexity.txt', vocab, device)
    save_perplexities_lstm(model, encoded_val, criterion, '2021101072_LM2_val_perplexity.txt', vocab, device)
    save_perplexities_lstm(model, encoded_test, criterion, '2021101072_LM2_test_perplexity.txt', vocab, device)

    import pickle

    with open('/Users/ashnadua/Desktop/2021101072_assignment1/data_store/data_store_lstm.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'vocab': vocab,
            'word_to_idx': word_to_idx,
            'encoded_train': encoded_train,
            'encoded_val': encoded_val,
            'encoded_test': encoded_test,
        }, f)

    print("Data saved successfully!")


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