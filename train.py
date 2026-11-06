import argparse, json, math, os
from collections import Counter
import torch
import torch.nn as nn
from common import tokenize, LM


def load_text(src):
    if src == "wikitext2":
        from datasets import load_dataset
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        return ["\n".join(d[s]["text"]) for s in ("train", "validation", "test")]
    t = open(src, encoding="utf-8").read()
    a, b = int(len(t) * 0.9), int(len(t) * 0.95)
    return [t[:a], t[a:b], t[b:]]


def make_seqs(ids, L):
    n = (len(ids) - 1) // L
    return ids[: n * L].view(n, L), ids[1 : n * L + 1].view(n, L)


@torch.no_grad()
def evaluate(model, x, y, dev, bs=256):
    model.eval()
    tot, cnt, hits = 0.0, 0, {1: 0, 3: 0, 5: 0}
    for i in range(0, len(x), bs):
        xb, yb = x[i : i + bs].to(dev), y[i : i + bs].to(dev)
        lo = model(xb)
        tot += nn.functional.cross_entropy(
            lo.reshape(-1, lo.size(-1)), yb.reshape(-1), reduction="sum"
        ).item()
        top = lo.topk(5, -1).indices
        for k in hits:
            hits[k] += (top[..., :k] == yb.unsqueeze(-1)).any(-1).sum().item()
        cnt += yb.numel()
    return {"loss": tot / cnt, "ppl": math.exp(tot / cnt),
            **{f"top{k}": hits[k] / cnt for k in hits}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext2", help="'wikitext2' or path to a .txt file")
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=35)
    ap.add_argument("--vocab", type=int, default=10000)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    tr_t, va_t, te_t = (tokenize(t) for t in load_text(a.data))
    itos = ["<unk>"] + [w for w, _ in Counter(tr_t).most_common(a.vocab - 1)]
    stoi = {w: i for i, w in enumerate(itos)}
    enc = lambda toks: torch.tensor([stoi.get(w, 0) for w in toks])
    (xtr, ytr), (xva, yva), (xte, yte) = (make_seqs(enc(t), a.seq_len) for t in (tr_t, va_t, te_t))
    print(f"device={dev} vocab={len(itos)} train_seqs={len(xtr)}")

    cfg = dict(emb=a.emb, hid=a.hid, layers=a.layers, kind=a.model, drop=0.3)
    model = LM(len(itos), **cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0)
    best, ckpt = float("inf"), f"{a.out}/{a.model}.pt"

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), a.bs):
            idx = perm[i : i + a.bs]
            xb, yb = xtr[idx].to(dev), ytr[idx].to(dev)
            loss = nn.functional.cross_entropy(model(xb).reshape(-1, len(itos)), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, xva, yva, dev)
        sched.step(v["loss"])
        print(f"epoch {ep}: train_loss={loss.item():.3f} val_ppl={v['ppl']:.1f} val_top3={v['top3']:.3f}")
        if v["loss"] < best:
            best = v["loss"]
            torch.save({"state": model.state_dict(), "itos": itos, "cfg": cfg}, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev)["state"])
    res = evaluate(model, xte, yte, dev)
    res["params"] = sum(p.numel() for p in model.parameters())
    json.dump(res, open(f"{a.out}/{a.model}.json", "w"), indent=2)
    print("TEST:", res)


if __name__ == "__main__":
    main()
