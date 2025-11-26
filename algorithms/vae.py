# ctext_vae_qna.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import nltk
nltk.download('punkt')


# local imports
from pathlib import Path
from utils.logmodule import logsetup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 32
EMB_DIM = 256
HID_DIM = 256
MAX_LEN = 128
BATCH = 16
LR = 5e-4
EPOCHS = 6

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode_text(text):
    t = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0)


class QADataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.questions = df["Question_Text"].tolist()
        self.answers = df["Answer_Text"].tolist()

    def __len__(self): return len(self.questions)

    def __getitem__(self, idx):
        q_ids, q_mask = encode_text(self.questions[idx])
        a_ids, a_mask = encode_text(self.answers[idx])
        return q_ids, q_mask, a_ids, a_mask


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(tokenizer.vocab_size, EMB_DIM)
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, batch_first=True)
        self.fc_mu = nn.Linear(HID_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HID_DIM, LATENT_DIM)

    def forward(self, x):
        h = self.emb(x)
        _, (h_n, _) = self.lstm(h)
        h_n = h_n[-1]       # last layer hidden state
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(tokenizer.vocab_size, EMB_DIM)
        self.fc = nn.Linear(LATENT_DIM, HID_DIM)
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, batch_first=True)
        self.out = nn.Linear(HID_DIM, tokenizer.vocab_size)

    def forward(self, z, target_ids):
        # z shape: (batch_size, LATENT_DIM)
        batch_size = z.size(0)
        
        # Initialize hidden state from latent z
        h0 = self.fc(z)  # (batch_size, HID_DIM)
        h0 = h0.unsqueeze(0)  # (1, batch_size, HID_DIM)
        c0 = torch.zeros_like(h0)  # (1, batch_size, HID_DIM)
        
        # Embed target sequences
        emb = self.emb(target_ids)  # (batch_size, seq_len, EMB_DIM)
        
        # Pass through LSTM
        out, _ = self.lstm(emb, (h0, c0))  # out: (batch_size, seq_len, HID_DIM)
        
        # Project to vocabulary
        logits = self.out(out)  # (batch_size, seq_len, vocab_size)
        return logits

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_q = Encoder()   # question encoder (conditioning)
        self.enc_a = Encoder()   # answer encoder
        self.dec = Decoder()

    def forward(self, q_ids, a_ids):
        mu_q, logvar_q = self.enc_q(q_ids)
        mu_a, logvar_a = self.enc_a(a_ids)

        # concatenate for conditioning
        mu = (mu_q + mu_a) / 2
        logvar = (logvar_q + logvar_a) / 2

        # reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # decode the answer
        logits = self.dec(z, a_ids)
        return logits, mu, logvar

def loss_fn(logits, target, mu, logvar):
    recon = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(logits.view(-1, logits.size(-1)), target.view(-1))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

def train(csv_path):
    ds = QADataset(csv_path)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = CVAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for q_ids, q_mask, a_ids, a_mask in dl:
            q_ids = q_ids.to(DEVICE)
            a_ids = a_ids.to(DEVICE)

            # Use only question encoder during generation
            mu_q, logvar_q = model.enc_q(q_ids)
            
            # For training, still use answer encoder
            mu_a, logvar_a = model.enc_a(a_ids)
            mu = (mu_q + mu_a) / 2
            logvar = (logvar_q + logvar_a) / 2
            
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            logits = model.dec(z, a_ids)
            loss = loss_fn(logits, a_ids, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss {total_loss/len(dl):.4f}")

    return model
@torch.no_grad()
def generate_answer(model, question_text, max_len=50):
    model.eval()
    
    q_ids, q_mask = encode_text(question_text)
    q_ids = q_ids.unsqueeze(0).to(DEVICE)

    # Encode question
    mu_q, logvar_q = model.enc_q(q_ids)
    z = mu_q
    
    h0 = model.dec.fc(z).unsqueeze(0)
    c0 = torch.zeros_like(h0).to(DEVICE)
    hidden = (h0, c0)

    cur_token = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)
    output_ids = []

    for _ in range(max_len):
        emb = model.dec.emb(cur_token)
        out, hidden = model.dec.lstm(emb, hidden)
        logits = model.dec.out(out)
        
        # Top-k sampling
        topk_logits, topk_indices = torch.topk(logits.squeeze(1), k=10)
        probs = torch.softmax(topk_logits, dim=-1)
        next_idx = torch.multinomial(probs, 1)
        next_id = topk_indices.gather(-1, next_idx)
        
        token_id = next_id.item()
        
        if token_id in [tokenizer.sep_token_id, tokenizer.pad_token_id, 0]:
            break
            
        output_ids.append(token_id)
        cur_token = next_id.view(1, 1)  # ← FIXED: use .view(1, 1) instead of .unsqueeze(0)

    return tokenizer.decode(output_ids, skip_special_tokens=True)

@torch.no_grad()
def evaluate_model(model, csv_path, num_samples=None):
    """Evaluate model on test set and compute BLEU scores"""
    model.eval()
    
    df = pd.read_csv(csv_path)
    questions = df["Question_Text"].tolist()
    references = df["Answer_Text"].tolist()
    
    if num_samples:
        questions = questions[:num_samples]
        references = references[:num_samples]
    
    bleu_scores = []
    smooth = SmoothingFunction()
    
    print(f"\nEvaluating on {len(questions)} samples...")
    
    for i, (question, reference) in enumerate(zip(questions, references)):
        # Generate answer
        generated = generate_answer(model, question)
        
        # Tokenize for BLEU (word-level)
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()
        
        # Calculate BLEU-4 score
        score = sentence_bleu(
            [ref_tokens], 
            gen_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth.method1
        )
        bleu_scores.append(score)
        
        # Print some examples
        if i < 5:
            print(f"\n--- Example {i+1} ---")
            print(f"Q: {question}")
            print(f"Reference: {reference}")
            print(f"Generated: {generated}")
            print(f"BLEU-4: {score:.4f}")
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\n{'='*50}")
    print(f"Average BLEU-4 Score: {avg_bleu:.4f}")
    print(f"{'='*50}")
    
    return avg_bleu, bleu_scores


if __name__ == "__main__":
    logger = logsetup("VAE_QA")

    csv_path = Path("datasets/Generative.csv")
    model = train(csv_path)
    question = "What is the reason of using stubs how is it helpful?"
    answer = generate_answer(model, question)
    logger.info(f"Q: {question}\nA: {answer}")

    
    # Save model
    torch.save(model.state_dict(), "cvae_model.pt")
    logger.info("Model saved!")
    
    # Evaluate on test set (or same set for now)
    avg_bleu, scores = evaluate_model(model, csv_path, num_samples=50)
    logger.info(f"Final BLEU-4 Score: {avg_bleu:.4f}")
