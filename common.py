import re
import torch
import torch.nn as nn


def tokenize(text):
    return re.findall(r"[a-z0-9']+|[.,!?;]", text.lower())


class LM(nn.Module):
    def __init__(self, vocab, emb=256, hid=512, layers=2, kind="lstm", drop=0.3):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = rnn(emb, hid, layers, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        return self.out(self.drop(h))


class GPT(nn.Module):
    """Small decoder-only transformer (causal self-attention, pre-LayerNorm)."""

    def __init__(self, vocab, emb=256, layers=8, heads=4, ff=1024, drop=0.1, max_len=35):
        super().__init__()
        self.max_ctx = max_len
        self.tok = nn.Embedding(vocab, emb)
        self.pos = nn.Embedding(max_len, emb)
        layer = nn.TransformerEncoderLayer(emb, heads, ff, drop, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(emb, vocab)

    def forward(self, x):
        T = x.size(1)
        h = self.drop(self.tok(x) + self.pos(torch.arange(T, device=x.device)))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.out(self.norm(h))


def build(vocab, cfg):
    cfg = dict(cfg)
    if cfg.get("kind") == "transformer":
        cfg.pop("kind")
        return GPT(vocab, **cfg)
    return LM(vocab, **cfg)


def load_ckpt(path, dev="cpu"):
    ck = torch.load(path, map_location=dev)
    model = build(len(ck["itos"]), ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["itos"], {w: i for i, w in enumerate(ck["itos"])}


@torch.no_grad()
def next_probs(model, stoi, text, temp=1.0, ctx=50):
    ctx = min(ctx, getattr(model, "max_ctx", ctx))
    ids = [stoi.get(w, 0) for w in tokenize(text)][-ctx:] or [0]
    logits = model(torch.tensor([ids]))[0, -1] / max(temp, 1e-3)
    logits[0] = float("-inf")  # never predict <unk>
    return torch.softmax(logits, -1)


def suggest(model, itos, stoi, text, k=5):
    p = next_probs(model, stoi, text)
    vals, idx = p.topk(k)
    return {itos[i]: float(v) for v, i in zip(vals, idx)}


def generate(model, itos, stoi, text, n=20, temp=0.8):
    out = text
    for _ in range(n):
        p = next_probs(model, stoi, out, temp)
        out += " " + itos[torch.multinomial(p, 1).item()]
    return out
