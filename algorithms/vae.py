# ctext_vae_qna.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 32
EMB_DIM = 256
HID_DIM = 256
MAX_LEN = 128
BATCH = 16
LR = 2e-4
EPOCHS = 20

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

# ---------------------------
# Dataset
# ---------------------------
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
        h0 = self.fc(z).unsqueeze(0)
        h0 = h0.repeat(1, target_ids.size(0), 1)
        emb = self.emb(target_ids)
        out, _ = self.lstm(emb, (h0, torch.zeros_like(h0)))
        logits = self.out(out)
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
        for q_ids, q_mask, a_ids, a_mask in dl:
            q_ids = q_ids.to(DEVICE)
            a_ids = a_ids.to(DEVICE)

            logits, mu, logvar = model(q_ids, a_ids)
            loss = loss_fn(logits, a_ids, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1} | Loss {loss.item():.4f}")

    return model


@torch.no_grad()
def generate_answer(model, question_text, max_len=80):
    q_ids, q_mask = encode_text(question_text)
    q_ids = q_ids.unsqueeze(0).to(DEVICE)

    # encode question
    mu_q, logvar_q = model.enc_q(q_ids)
    std = torch.exp(0.5 * logvar_q)
    z = mu_q + torch.randn_like(std) * std

    # start token
    cur = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)

    output_ids = []

    for _ in range(max_len):
        logits = model.dec(z, cur)
        next_id = logits[:, -1].argmax(-1).unsqueeze(0)
        output_ids.append(next_id.item())

        cur = torch.cat([cur, next_id], dim=1)
        if next_id.item() == tokenizer.sep_token_id:
            break

    return tokenizer.decode(output_ids)


if __name__ == "__main__":
    model = train("datasets/Generative.csv")
    question = "What is the reason of using stubs how is it helpful?"
    answer = generate_answer(model, question)
    print(f"Q: {question}\nA: {answer}")
