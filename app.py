import os
import torch
import gradio as gr
from common import load_ckpt, suggest, generate

RUNS = "runs103" if os.path.exists("runs103") else "runs"
MODELS = {k: load_ckpt(f"{RUNS}/{k}.pt") for k in ("lstm", "gru", "transformer") if os.path.exists(f"{RUNS}/{k}.pt")}
GPT = {}


def gpt_load():
    if not GPT:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        vocab = [tok.decode([i]) for i in range(model.config.vocab_size)]
        ok = torch.tensor([v.startswith(" ") and v[1:].isalpha() for v in vocab])
        GPT.update(tok=tok, model=model, ok=ok)
    return GPT["tok"], GPT["model"], GPT["ok"]


@torch.no_grad()
def gpt_predict(text, temp, n):
    tok, model, ok = gpt_load()
    ids = tok(text.rstrip(), return_tensors="pt").input_ids[:, -200:]
    logits = model(ids).logits[0, -1].clone()
    logits[~ok] = float("-inf")  # only allow tokens that start a new word
    vals, idx = torch.softmax(logits, -1).topk(5)
    top = {tok.decode([int(j)]).strip(): float(p) for p, j in zip(vals, idx)}
    first = torch.multinomial(torch.softmax(logits / temp, -1), 1)
    ids = torch.cat([ids, first.view(1, 1)], dim=1)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=int(n),
                         do_sample=True, temperature=temp, top_k=50, pad_token_id=tok.eos_token_id)
    return top, tok.decode(out[0], skip_special_tokens=True)


def predict(text, which, temp, n_words):
    if not text.strip():
        return {}, ""
    if which == "distilgpt2 (pretrained)":
        return gpt_predict(text, temp, n_words)
    model, itos, stoi = MODELS[which]
    return suggest(model, itos, stoi, text), generate(model, itos, stoi, text, int(n_words), temp)


with gr.Blocks(title="Next Word Predictor") as demo:
    gr.Markdown("# Next Word Predictor\nCompare LSTM and GRU (trained from scratch) with pretrained DistilGPT-2.")
    text = gr.Textbox(label="Your text", lines=2, placeholder="the history of the")
    with gr.Row():
        which = gr.Radio(list(MODELS) + ["distilgpt2 (pretrained)"], value=list(MODELS)[0], label="Model")
        temp = gr.Slider(0.3, 1.5, 0.8, label="Temperature (generation)")
        n_words = gr.Slider(5, 50, 20, step=1, label="Words to generate")
    label = gr.Label(num_top_classes=5, label="Top-5 next words")
    gen = gr.Textbox(label="Generated continuation")
    for ev in (text.change, which.change, temp.release, n_words.release):
        ev(predict, [text, which, temp, n_words], [label, gen])

if __name__ == "__main__":
    demo.launch()

