"""Quick check of sequence length distribution in data splits."""
import os, sys
from datasets import load_from_disk

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sgtm/coarse"
seqs = []
for split in ["forget", "retain", "adjacent"]:
    path = os.path.join(data_dir, split, "train")
    if not os.path.isdir(path):
        continue
    ds = load_from_disk(path)
    lengths = [len(r["sequence"]) for r in ds]
    seqs.extend(lengths)
    lengths.sort()
    n = len(lengths)
    print(f"{split}: {n} seqs, median={lengths[n//2]}, p95={lengths[int(n*0.95)]}, max={lengths[-1]}")

seqs.sort()
n = len(seqs)
print(f"\nAll: {n} seqs")
print(f"Min={seqs[0]}, Median={seqs[n//2]}, P95={seqs[int(n*0.95)]}, P99={seqs[int(n*0.99)]}, Max={seqs[-1]}")
print(f"Over 512: {sum(1 for s in seqs if s > 512)}, Over 1000: {sum(1 for s in seqs if s > 1000)}")
