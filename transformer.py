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
from save_perplexities import save_perplexities_transformer

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerDataset(Dataset):
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


def pos_encoding(num_tokens, n_dim):
    pos_enc = np.zeros((num_tokens, n_dim))
    positions = np.arange(num_tokens)[:, np.newaxis]
    div_term = np.exp(np.arange(0, n_dim, 2) * -(np.log(10000.0) / n_dim))
    pos_enc[:, 0::2] = np.sin(positions * div_term)
    pos_enc[:, 1::2] = np.cos(positions * div_term)
    return torch.tensor(pos_enc, dtype=torch.float)


class TransformerDecoder(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.positional_encoding = pos_encoding(200, embedding_dim).to(device)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=num_layers)
        
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, tgt, tgt_mask=None):
        tgt = self.embedding(tgt) + self.positional_encoding[:tgt.size(1)]
        tgt = self.layer_norm(tgt)
        output = self.transformer_decoder(tgt, memory=tgt, tgt_mask=tgt_mask)  # Use tgt as memory
        output = self.fc_out(self.dropout(output))
        return output
    

def test_model_transformer(model, val_loader, criterion, pad_idx):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output.view(-1, output.shape[2]), y.view(-1)) 
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    return avg_val_loss, val_perplexity


def train_model_transformer(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, patience=2, pad_idx=0):
    model.to(device)
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output.view(-1, output.shape[2]), y.view(-1)) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_train_loss))

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Train Perplexity: {perplexity:.4f}')

        avg_val_loss, val_perplexity = test_model_transformer(model, val_loader, criterion, pad_idx)
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Perplexity: {val_perplexity:.4f}')

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), '2021101072_LM3.pt')
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
    with open('/Users/ashnadua/Desktop/2021101072_assignment1/data_store/data_store_transformer.pkl', 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    vocab = data['vocab']
    word_to_idx = data['word_to_idx']
    encoded_train = data['encoded_train']
    encoded_val = data['encoded_val']
    encoded_test = data['encoded_test']

    # Recreate datasets and loaders
    train_dataset = TransformerDataset(encoded_train)
    val_dataset = TransformerDataset(encoded_val)
    test_dataset = TransformerDataset(encoded_test)

    pad_idx = word_to_idx['<PAD>']

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)

    print("Data loaded successfully!")

    vocab_size = len(vocab)
    embedding_dim = 300
    num_heads = 10
    num_layers = 2
    hidden_dim = 300
    dropout = 0.1

    model_new = TransformerDecoder(embedding=embeddings, vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model_new.load_state_dict(torch.load('/Users/ashnadua/Desktop/2021101072_assignment1/models/2021101072_LM3.pt', map_location=device))

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model_new.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, factor=0.1)

    loss, perplexity = test_model_transformer(model_new, val_loader, criterion, pad_idx)
    print(f'\nVal Loss: {loss}')
    print(f'Val Perplexity: {perplexity}\n')

    loss, perplexity = test_model_transformer(model_new, test_loader, criterion, pad_idx)
    print(f'\nTest Loss: {loss}')
    print(f'Test Perplexity: {perplexity}')



def run_model():
    print("Training model...")
    with open('/Users/ashnadua/Desktop/2021101072_assignment1/Auguste_Maquet.txt', 'r') as f:
        corpus = f.read()
        
    corpus = preprocess(corpus) 

    sentences, word_sentences = tokenize(corpus, 2)

    train_sentences, val_sentences, test_sentences = train_val_test_split(word_sentences)

    print("Train size:", len(train_sentences))
    print("Validation size:", len(val_sentences))
    print("Test size:", len(test_sentences))

    glove = create_glove_embeddings('/Users/ashnadua/Desktop/2021101072_assignment1/glove/glove.6B.300d.txt')
    
    embeddings, encoded_train, encoded_val, encoded_test, word_to_idx, vocab = create_embeddings_and_encode(train_sentences, val_sentences, test_sentences, glove)

    pad_idx = word_to_idx['<PAD>']

    train_dataset = TransformerDataset(encoded_train)
    val_dataset = TransformerDataset(encoded_val)
    test_dataset = TransformerDataset(encoded_test)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda batch: collate_fn(batch, pad_idx), shuffle=True)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    vocab_size = len(vocab)
    embedding_dim = 300
    num_heads = 10
    num_layers = 2
    hidden_dim = 300
    dropout = 0.1

    model = TransformerDecoder(embedding=embeddings, vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, factor=0.1)

    model = train_model_transformer(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=10, patience=3, pad_idx=pad_idx)

    loss, perplexity = test_model_transformer(model, train_loader, criterion, pad_idx)
    print(f'\nTrain Loss: {loss}')
    print(f'Train Perplexity: {perplexity}\n')

    loss, perplexity = test_model_transformer(model, val_loader, criterion, pad_idx)
    print(f'\nVal Loss: {loss}')
    print(f'Val Perplexity: {perplexity}\n')

    loss, perplexity = test_model_transformer(model, test_loader, criterion, pad_idx)
    print(f'\nTest Loss: {loss}')
    print(f'Test Perplexity: {perplexity}')

    save_perplexities_transformer(model, encoded_train, criterion, '2021101072_LM3_train_perplexity.txt', vocab, pad_idx, device)
    save_perplexities_transformer(model, encoded_val, criterion, '2021101072_LM3_val_perplexity.txt', vocab, pad_idx, device)
    save_perplexities_transformer(model, encoded_test, criterion, '2021101072_LM3_test_perplexity.txt', vocab, pad_idx, device)

    with open('/Users/ashnadua/Desktop/2021101072_assignment1/data_store/data_store_transformer.pkl', 'wb') as f:
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