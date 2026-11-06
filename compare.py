import glob, json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "runs"
print("| Model | Params | Test perplexity | Top-1 | Top-3 | Top-5 |")
print("|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{d}/*.json")):
    r = json.load(open(f))
    print(f"| {os.path.basename(f)[:-5].upper()} | {r['params']/1e6:.1f}M | {r['ppl']:.1f} | "
          f"{r['top1']:.1%} | {r['top3']:.1%} | {r['top5']:.1%} |")
