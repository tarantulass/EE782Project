# ctext_vae_qna.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os, sys

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import nltk
nltk.download('punkt')

from pathlib import Path
import logging

def logsetup(log_filename=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_filename:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 64  # Increased for more capacity
EMB_DIM = 256    # Larger embeddings
HID_DIM = 512    # Much larger hidden state
MAX_LEN = 64
BATCH = 16       # Smaller batch for better gradients
LR = 5e-4        # Lower learning rate
EPOCHS = 150     # More epochs

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
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(HID_DIM * 2, LATENT_DIM)
        self.fc_logvar = nn.Linear(HID_DIM * 2, LATENT_DIM)

    def forward(self, x):
        h = self.emb(x)
        _, (h_n, _) = self.lstm(h)
        # Concatenate forward and backward
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(tokenizer.vocab_size, EMB_DIM)

        # FIXED: q_encoding is LATENT_DIM (64), not HID_DIM*2
        self.fc_init = nn.Linear(LATENT_DIM * 2, HID_DIM)  # z + q_encoding (both 64-dim)

        # Concatenate both z AND q_encoding to every timestep
        self.lstm = nn.LSTM(EMB_DIM + LATENT_DIM * 2, HID_DIM, batch_first=True, num_layers=2, dropout=0.3)
        self.out = nn.Linear(HID_DIM, tokenizer.vocab_size)

    def forward(self, z, q_encoding, target_ids):
        batch_size = z.size(0)
        seq_len = target_ids.size(1)

        # Initialize hidden state from z + question (both 64-dim)
        h0 = torch.tanh(self.fc_init(torch.cat([z, q_encoding], dim=1)))
        h0 = h0.unsqueeze(0).repeat(2, 1, 1)  # 2 layers
        c0 = torch.zeros_like(h0)

        # Embed target tokens
        emb = self.emb(target_ids)  # (batch, seq_len, EMB_DIM=256)

        # Concatenate BOTH z and q_encoding to every time step
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, 64)
        q_expanded = q_encoding.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, 64)
        emb = torch.cat([emb, z_expanded, q_expanded], dim=2)  # (batch, seq_len, 256+64+64=384)

        out, _ = self.lstm(emb, (h0, c0))
        logits = self.out(out)
        return logits

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_q = Encoder()
        self.enc_a = Encoder()
        self.dec = Decoder()

    def forward(self, q_ids, a_ids):
        mu_q, logvar_q = self.enc_q(q_ids)
        mu_a, logvar_a = self.enc_a(a_ids)

        # Keep question encoding separate for decoder conditioning
        q_encoding = mu_q.detach()  # Use as context, don't backprop through it in decoder

        # Only use answer encoder for latent space
        mu = mu_a
        logvar = logvar_a

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        logits = self.dec(z, q_encoding, a_ids)
        return logits, mu, logvar, mu_q

def loss_fn(logits, target, mu, logvar, beta=0.5):
    recon = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
        logits.reshape(-1, logits.size(-1)),  # Use reshape instead of view
        target.reshape(-1)                     # Use reshape instead of view
    )
    # KL annealing - start with more weight
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

def train(csv_path):
    ds = QADataset(csv_path)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = CVAE().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        # KL annealing schedule
        beta = min(1.0, 0.1 + (epoch / EPOCHS) * 0.9)  # 0.1 → 1.0

        for q_ids, q_mask, a_ids, a_mask in dl:
            q_ids = q_ids.to(DEVICE)
            a_ids = a_ids.to(DEVICE)

            # Teacher forcing
            decoder_input = a_ids[:, :-1]
            decoder_target = a_ids[:, 1:]

            mu_q, logvar_q = model.enc_q(q_ids)
            mu_a, logvar_a = model.enc_a(a_ids)

            q_encoding = mu_q
            mu = mu_a
            logvar = logvar_a

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            logits = model.dec(z, q_encoding, decoder_input)

            loss, recon, kl = loss_fn(logits, decoder_target, mu, logvar, beta)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss {total_loss/len(dl):.4f} | Recon {total_recon/len(dl):.4f} | KL {total_kl/len(dl):.4f} | Beta {beta:.3f}")

    return model

@torch.no_grad()
def generate_answer(model, question_text, max_len=50, temperature=0.7):
    model.eval()

    q_ids, q_mask = encode_text(question_text)
    q_ids = q_ids.unsqueeze(0).to(DEVICE)

    # Get question encoding (64-dim)
    mu_q, logvar_q = model.enc_q(q_ids)
    q_encoding = mu_q

    # Sample z from prior (64-dim)
    z = torch.randn(1, LATENT_DIM).to(DEVICE) * 0.5

    # Initialize decoder
    h0 = torch.tanh(model.dec.fc_init(torch.cat([z, q_encoding], dim=1)))
    h0 = h0.unsqueeze(0).repeat(2, 1, 1)  # 2 layers
    c0 = torch.zeros_like(h0).to(DEVICE)
    hidden = (h0, c0)

    cur_token = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)
    output_ids = []

    for step in range(max_len):
        emb = model.dec.emb(cur_token)

        # Concatenate BOTH z and q_encoding to input at each step
        z_step = z.unsqueeze(1)  # (1, 1, 64)
        q_step = q_encoding.unsqueeze(1)  # (1, 1, 64)
        emb = torch.cat([emb, z_step, q_step], dim=2)  # (1, 1, 256+64+64=384)

        out, hidden = model.dec.lstm(emb, hidden)
        logits = model.dec.out(out).squeeze(1)

        # Nucleus sampling (top-p)
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold (nucleus)
        sorted_indices_to_remove = cumulative_probs > 0.9
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        token_id = next_id.item()

        if token_id in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
            break

        output_ids.append(token_id)
        cur_token = next_id

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    return decoded


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
        generated = generate_answer(model, question)

        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        score = sentence_bleu(
            [ref_tokens],
            gen_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth.method1
        )
        bleu_scores.append(score)

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
    logger = logsetup("VAE_QA.log")

    csv_path = Path("generate.csv")

    print("Starting training...")
    model = train(csv_path)

    print("\nGenerating sample answer...")
    question = "What is the reason of using stubs how is it helpful?"
    answer = generate_answer(model, question)
    logger.info(f"Q: {question}\nA: {answer}")

    # Save model
    torch.save(model.state_dict(), "cvae_model.pt")
    logger.info("Model saved!")

    # Evaluate on ENTIRE dataset
    avg_bleu, scores = evaluate_model(model, csv_path, num_samples=None)
    logger.info(f"Final BLEU-4 Score: {avg_bleu:.4f}")