# Next Word Predictor: LSTM vs GRU

Word-level next-word prediction with recurrent networks in PyTorch, trained on WikiText-2,
with proper evaluation (perplexity, top-k accuracy) and a live Gradio demo.

Based on the idea in [Bharath-K3/Next-Word-Prediction-with-NLP-and-Deep-Learning](https://github.com/Bharath-K3/Next-Word-Prediction-with-NLP-and-Deep-Learning).

## What's different from the original
- Larger dataset (WikiText-2) instead of one short book
- Held-out validation/test split, perplexity and top-1/3/5 accuracy
- LSTM vs GRU comparison under identical settings
- Interactive demo with top-5 suggestions, temperature and text generation

## Run
```bash
pip install -r requirements.txt
python train.py --model lstm --epochs 8
python train.py --model gru  --epochs 8
python compare.py      # prints the results table
python app.py          # launches the demo
```
Use `--data path/to/file.txt` to train on your own text. A free Colab/Kaggle GPU is enough.

## Results
Paste the output of `python compare.py` here, then add 5-10 example predictions,
2-3 failure cases (rare words, long-range context), and a screenshot of the demo.
## Results
Trained for 8 epochs on WikiText-2 (10k-word vocabulary, sequence length 35, same hyperparameters for both models).

| Model | Params | Test perplexity | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| GRU | 10.4M | 127.6 | 22.1% | 38.0% | 45.1% |
| LSTM | 11.4M | 107.5 | 23.8% | 40.0% | 47.0% |

The LSTM generalized better in this setup: its validation perplexity kept improving through epoch 8, while the GRU plateaued earlier. Caveats: single run per model, and the LSTM has about 9% more parameters.

## Limitations
- Closed 10k-word vocabulary; unseen words map to `<unk>`
- Trained on Wikipedia text, so suggestions follow that style
- A transformer or subword tokenizer would handle rare words and long context better
