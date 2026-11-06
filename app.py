import os
import gradio as gr
from common import load_ckpt, suggest, generate

MODELS = {k: load_ckpt(f"runs/{k}.pt") for k in ("lstm", "gru") if os.path.exists(f"runs/{k}.pt")}


def predict(text, which, temp, n_words):
    model, itos, stoi = MODELS[which]
    if not text.strip():
        return {}, ""
    return suggest(model, itos, stoi, text), generate(model, itos, stoi, text, int(n_words), temp)


with gr.Blocks(title="Next Word Predictor") as demo:
    gr.Markdown("# Next Word Predictor\nType a few words and see the model's top suggestions.")
    text = gr.Textbox(label="Your text", lines=2, placeholder="the history of the")
    with gr.Row():
        which = gr.Radio(list(MODELS), value=list(MODELS)[0], label="Model")
        temp = gr.Slider(0.3, 1.5, 0.8, label="Temperature (generation)")
        n_words = gr.Slider(5, 50, 20, step=1, label="Words to generate")
    label = gr.Label(num_top_classes=5, label="Top-5 next words")
    gen = gr.Textbox(label="Generated continuation")
    for ev in (text.change, which.change, temp.release, n_words.release):
        ev(predict, [text, which, temp, n_words], [label, gen])

if __name__ == "__main__":
    demo.launch()
