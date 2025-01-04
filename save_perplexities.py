import torch

def save_perplexities(model, sentences, embeddings, criterion, filename, idx_to_word, device, n=5):
    model.eval()
    total_loss = 0
    all_sentences = []
    perplexity_scores = []

    with torch.no_grad():
        for sentence in sentences:
            sentence_loss = 0
            sentence_length = 0
            ngrams = []
            targets = []

            for i in range(len(sentence) - n):
                context_indices = sentence[i:i + n]
                target_index = sentence[i + n]

                context_embeddings = torch.cat([embeddings[idx] for idx in context_indices], dim=0).float()

                ngrams.append(context_embeddings)
                targets.append(target_index)

            for j in range(len(ngrams)):
                ngram = ngrams[j].to(device) 
                target_word = torch.tensor(targets[j], dtype=torch.long).to(device)  

                outputs = model(ngram.unsqueeze(0))  
                loss = criterion(outputs, target_word.unsqueeze(0))  
                sentence_loss += loss.item()
                sentence_length += 1

            avg_loss_per_sentence = sentence_loss / sentence_length
            sentence_perplexity = torch.exp(torch.tensor(avg_loss_per_sentence)).item()
            perplexity_scores.append(sentence_perplexity)

            sentence_words = [idx_to_word[idx] for idx in sentence]
            full_sentence = " ".join(sentence_words)
            all_sentences.append(full_sentence)

        avg_perplexity = sum(perplexity_scores) / len(perplexity_scores)

    with open(filename, 'w') as f:
        for i, sentence in enumerate(all_sentences):
            f.write(f"{sentence}\t{perplexity_scores[i]}\n")
        
        f.write(f"Average\t{avg_perplexity}\n")

    return avg_perplexity

def save_perplexities_lstm(model, sentences, criterion, filename, idx_to_word, device):
    model.eval()
    total_loss = 0
    all_sentences = []
    perplexity_scores = []

    with torch.no_grad():
        for sentence in sentences:
            sentence_loss = 0
            sentence_length = 0
            input_indices = sentence[:-1]
            target_indices = sentence[1:]
            
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
            targets = torch.tensor(target_indices, dtype=torch.long).to(device)
            outputs, _ = model(input_tensor)

            for i in range(outputs.shape[1]):
                output = outputs[0, i]
                target_word = targets[i]

                loss = criterion(output.unsqueeze(0), target_word.unsqueeze(0))
                sentence_loss += loss.item()
                sentence_length += 1

            avg_loss_per_sentence = sentence_loss / sentence_length
            sentence_perplexity = torch.exp(torch.tensor(avg_loss_per_sentence)).item()
            perplexity_scores.append(sentence_perplexity)

            sentence_words = [idx_to_word[idx] for idx in sentence]
            full_sentence = " ".join(sentence_words)
            all_sentences.append(full_sentence)

        avg_perplexity = sum(perplexity_scores) / len(perplexity_scores)

    with open(filename, 'w') as f:
        for i, sentence in enumerate(all_sentences):
            f.write(f"{sentence}\t{perplexity_scores[i]:.4f}\n")
        
        f.write(f"Average\t{avg_perplexity:.4f}\n")

    return avg_perplexity


def save_perplexities_transformer(model, sentences, criterion, filename, idx_to_word, pad_idx, device):
    model.eval()
    total_loss = 0
    all_sentences = []
    perplexity_scores = []

    with torch.no_grad():
        for sentence in sentences:
            sentence_loss = 0
            sentence_length = 0
            input_indices = sentence[:-1]
            target_indices = sentence[1:]

            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
            targets = torch.tensor(target_indices, dtype=torch.long).to(device)

            outputs = model(input_tensor)

            for i in range(outputs.shape[1]):
                output = outputs[0, i]
                target_word = targets[i]

                if target_word != pad_idx:
                    loss = criterion(output.unsqueeze(0), target_word.unsqueeze(0))
                    sentence_loss += loss.item()
                    sentence_length += 1

            if sentence_length > 0:
                avg_loss_per_sentence = sentence_loss / sentence_length
                sentence_perplexity = torch.exp(torch.tensor(avg_loss_per_sentence)).item()
            else:
                sentence_perplexity = float('inf')  # handle empty sentences (unlikely, but a safeguard)

            perplexity_scores.append(sentence_perplexity)
            sentence_words = [idx_to_word[idx] for idx in sentence]
            full_sentence = " ".join(sentence_words)
            all_sentences.append(full_sentence)

        avg_perplexity = sum(perplexity_scores) / len(perplexity_scores)

    with open(filename, 'w') as f:
        for i, sentence in enumerate(all_sentences):
            f.write(f"{sentence}\t{perplexity_scores[i]:.4f}\n")
        
        f.write(f"Average\t{avg_perplexity:.4f}\n")

    return avg_perplexity
