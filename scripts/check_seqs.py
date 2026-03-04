"""Quick check of sequence length distribution in data splits."""
import os, sys

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sgtm/coarse"
seqs = []
for f in os.listdir(data_dir):
    if not f.endswith(".tsv"):
        continue
    for line in open(os.path.join(data_dir, f)):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            seqs.append(len(parts[1]))

seqs.sort()
n = len(seqs)
print(f"Files in: {data_dir}")
print(f"Count: {n}")
print(f"Min: {seqs[0]}, Median: {seqs[n//2]}, P95: {seqs[int(n*0.95)]}, P99: {seqs[int(n*0.99)]}, Max: {seqs[-1]}")
print(f"Over 512: {sum(1 for s in seqs if s > 512)}, Over 1000: {sum(1 for s in seqs if s > 1000)}")
