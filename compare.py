import glob, json, os
print("| Model | Params | Test perplexity | Top-1 | Top-3 | Top-5 |")
print("|---|---|---|---|---|---|")
for f in sorted(glob.glob("runs/*.json")):
    r = json.load(open(f))
    print(f"| {os.path.basename(f)[:-5].upper()} | {r['params']/1e6:.1f}M | {r['ppl']:.1f} | "
          f"{r['top1']:.1%} | {r['top3']:.1%} | {r['top5']:.1%} |")
