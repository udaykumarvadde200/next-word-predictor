# Next Word Predictor: LSTM vs GRU vs Transformer

Word-level next-word prediction in PyTorch, comparing an LSTM, a GRU and a small GPT-style transformer
under the same data, vocabulary and test set, with a live Gradio demo that also compares against pretrained DistilGPT-2.

Inspired by [Bharath-K3/Next-Word-Prediction-with-NLP-and-Deep-Learning](https://github.com/Bharath-K3/Next-Word-Prediction-with-NLP-and-Deep-Learning).

## What's different from the original
- Much larger datasets (WikiText-2, then a subset of WikiText-103) instead of one short book
- Validation/test splits with perplexity and top-1/3/5 accuracy
- LSTM vs GRU vs transformer comparison at about 11M parameters each
- Demo with top-5 suggestions, temperature slider, text generation, and a pretrained DistilGPT-2 option for comparison

## Results
Test set: WikiText test split, 10k-word vocabulary, sequence length 35, best checkpoint by validation loss.

**WikiText-103 subset** (first 400,000 lines of the training split, 4 epochs, batch size 256)

| Model | Params | Test perplexity | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| GRU | 10.4M | 82.0 | 24.9% | 41.3% | 48.4% |
| LSTM | 11.4M | 71.0 | 26.1% | 42.7% | 49.8% |
| Transformer (8 layers, 4 heads) | 11.5M | 57.7 | 27.9% | 45.0% | 52.2% |

**WikiText-2** (12 epochs, RNNs only)

| Model | Params | Test perplexity | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| GRU | 10.4M | 126.2 | 22.5% | 38.3% | 45.3% |
| LSTM | 11.4M | 108.7 | 24.1% | 40.2% | 47.1% |

Findings: the transformer reached about 19% lower perplexity than the LSTM (57.7 vs 71.0) at nearly the same parameter count, and the LSTM beat the GRU on both datasets. More training data cut the LSTM's perplexity from 108.7 to 71.0.
Caveats: one run per model (no seed variance); the transformer used a different learning rate, warmup and dropout than the RNNs, and none of the models were tuned; validation perplexity was still decreasing at epoch 4, so all results are for a fixed short budget.

## Failure cases and lessons
- **Low confidence and topic drift:** with no pretraining, the models give low probabilities to many plausible words and generated text wanders off topic. The prompt "a king named alexander the" needs world knowledge (alexander the great) that a small from-scratch model only partly has.
- **Bug found and fixed:** my first tokenizer stripped digits, so "19th century" became "th century". Numbers are now kept.
- **Demo bug found and fixed:** DistilGPT-2 predicts subword tokens, so a prompt ending in "the" produced word fragments. The demo now restricts its suggestions to tokens that start a new word.

## Run
```bash
pip install -r requirements.txt
python train.py --data wikitext103 --model lstm --epochs 4 --bs 256 --out runs103
python train.py --data wikitext103 --model gru --epochs 4 --bs 256 --out runs103
python train.py --data wikitext103 --model transformer --layers 8 --heads 4 --ff 1024 --drop 0.1 --lr 1e-3 --warmup 500 --epochs 4 --bs 256 --out runs103
python compare.py runs103
python app.py
```
Use `--data wikitext2` (default) or `--data path/to/file.txt` for other data. The demo loads `runs103/` if it exists, otherwise `runs/`.
Model files (`.pt`) are not in the repo; train them with the commands above.

## Limitations and next steps
- Closed 10k-word vocabulary: unseen words become `<unk>`
- DistilGPT-2 predicts subword tokens, so its perplexity is not comparable to these word-level numbers
- Next steps: multiple seeds with mean and standard deviation, longer training, subword tokenizer

## Demo
![demo](screenshot.png)

![failure case](failure.png)
