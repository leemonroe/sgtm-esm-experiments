"""
SGTM data pipeline: download Swiss-Prot, split virus/non-virus data into
forget/adjacent/ambiguous/retain categories, and save as HuggingFace datasets.
"""

import argparse
import gzip
import os
import re
import urllib.request
from typing import Dict, List, Tuple

import torch
from datasets import Dataset

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 30
MAX_LENGTH = 1022

SWISSPROT_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/"
    "complete/uniprot_sprot.fasta.gz"
)


# ---------------------------------------------------------------------------
# Download & parse
# ---------------------------------------------------------------------------

def download_swissprot(output_path: str) -> str:
    """Download Swiss-Prot FASTA (gzipped) and return path to the .fasta.gz file."""
    output_path = str(output_path)
    if os.path.exists(output_path):
        print(f"Swiss-Prot already downloaded: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Downloading Swiss-Prot to {output_path} ...")
    urllib.request.urlretrieve(SWISSPROT_URL, output_path)
    print(f"Download complete ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return output_path


def parse_fasta_gz(path: str, return_headers: bool = False):
    """Parse a gzipped FASTA file.

    Args:
        path: Path to .fasta.gz file.
        return_headers: If True, return (sequences, headers) tuple.

    Returns:
        List of sequences, or (sequences, headers) if return_headers=True.
    """
    sequences = []
    headers = []
    current_seq_parts: List[str] = []
    current_header = ""

    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq_parts:
                    sequences.append("".join(current_seq_parts))
                    headers.append(current_header)
                    current_seq_parts = []
                current_header = line
            else:
                current_seq_parts.append(line)
        if current_seq_parts:
            sequences.append("".join(current_seq_parts))
            headers.append(current_header)

    print(f"Parsed {len(sequences):,} sequences from {path}")
    if return_headers:
        return sequences, headers
    return sequences


def load_virus_data(raw_dir: str) -> Tuple[List[str], List[str]]:
    """Load human-infecting and non-human virus sequences from TSV files."""
    human_path = os.path.join(raw_dir, "virus_human.tsv")
    nonhuman_path = os.path.join(raw_dir, "virus_nonhuman.tsv")

    def read_tsv_sequences(path: str) -> List[str]:
        seqs = []
        with open(path) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    seqs.append(parts[3])  # Sequence column
        return seqs

    human_seqs = read_tsv_sequences(human_path)
    nonhuman_seqs = read_tsv_sequences(nonhuman_path)
    print(f"Loaded virus data: {len(human_seqs):,} human, {len(nonhuman_seqs):,} non-human")
    return human_seqs, nonhuman_seqs


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def is_valid_sequence(seq: str) -> bool:
    """Check standard AAs only and length within bounds."""
    return (
        MIN_LENGTH <= len(seq) <= MAX_LENGTH
        and all(aa in VALID_AAS for aa in seq)
    )


# Regex to extract OS= field from UniProt FASTA headers
_OS_PATTERN = re.compile(r"OS=(.+?)(?:\s+OX=|\s+GN=|\s+PE=|\s*$)")

# Keywords indicating a viral organism (case-insensitive match on OS= field)
_VIRAL_KEYWORDS = ("virus", "phage", "viridae", "viral", "virales")


def _is_viral_header(header: str) -> bool:
    """Check if a Swiss-Prot FASTA header indicates a viral organism.

    Swiss-Prot headers have the format:
        >sp|P12345|NAME_ORGANISM Description OS=Organism name OX=TaxID ...

    We check the OS= field for viral keywords.
    """
    match = _OS_PATTERN.search(header)
    if not match:
        return False
    organism = match.group(1).lower()
    return any(kw in organism for kw in _VIRAL_KEYWORDS)


def filter_swissprot(
    swissprot_seqs: List[str],
    virus_seqs_to_exclude: set,
    headers: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """Filter Swiss-Prot: valid AAs, length 30-1022, deduplicate against virus data.

    Also identifies viral proteins by FASTA header keyword matching.
    These would contaminate the retain set if not separated.

    NOTE: The 8M experiments did NOT have this viral header filter. Swiss-Prot
    contains ~16K viral proteins beyond our virus TSV files, which leaked into
    the retain set in the 8M runs. This is fixed here for 35M onwards.

    Returns:
        (retain_seqs, viral_swissprot_seqs): Non-viral sequences for retain set,
        and viral Swiss-Prot sequences to route to the ambiguous set.
    """
    retain = []
    viral_swissprot = []
    seen = set()
    n_viral_header = 0

    for i, seq in enumerate(swissprot_seqs):
        if not is_valid_sequence(seq):
            continue
        if seq in virus_seqs_to_exclude:
            continue
        if seq in seen:
            continue
        seen.add(seq)

        # Check if this is a viral protein by header
        if headers and _is_viral_header(headers[i]):
            viral_swissprot.append(seq)
            n_viral_header += 1
        else:
            retain.append(seq)

    total_filtered = len(retain) + len(viral_swissprot)
    print(f"Swiss-Prot after filtering: {total_filtered:,} / {len(swissprot_seqs):,}")
    if n_viral_header > 0:
        print(f"  Viral proteins detected by header: {n_viral_header:,} (routed to ambiguous)")
    print(f"  Non-viral retain sequences: {len(retain):,}")
    return retain, viral_swissprot


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _split_three(seqs: List[str], ratios: Tuple[float, float, float], rng):
    """Split a list into three parts according to ratios."""
    n = len(seqs)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = [seqs[i] for i in indices[:n_train]]
    val = [seqs[i] for i in indices[n_train : n_train + n_val]]
    test = [seqs[i] for i in indices[n_train + n_val :]]
    return train, val, test


def _seqs_to_dataset(seqs: List[str]) -> Dataset:
    return Dataset.from_dict({
        "sequence": seqs,
        "length": [len(s) for s in seqs],
    })


def prepare_datasets(
    human_seqs: List[str],
    nonhuman_seqs: List[str],
    swissprot_seqs: List[str],
    output_dir: str,
    seed: int = 42,
    swissprot_headers: List[str] = None,
) -> Dict[str, Dataset]:
    """
    Split data into SGTM categories and save as HuggingFace datasets.

    Categories:
      - forget:    90% of virus_human
      - adjacent:  90% of virus_nonhuman
      - ambiguous: 10% of both virus sets (the "default" partition)
      - retain:    filtered Swiss-Prot remainder
    """
    import random
    rng = random.Random(seed)

    # Filter virus sequences too
    human_seqs = [s for s in human_seqs if is_valid_sequence(s)]
    nonhuman_seqs = [s for s in nonhuman_seqs if is_valid_sequence(s)]
    print(f"Valid virus sequences: {len(human_seqs)} human, {len(nonhuman_seqs)} non-human")

    # Shuffle before splitting
    human_idx = list(range(len(human_seqs)))
    nonhuman_idx = list(range(len(nonhuman_seqs)))
    rng.shuffle(human_idx)
    rng.shuffle(nonhuman_idx)

    # 90/10 split for each virus set
    human_90 = int(len(human_seqs) * 0.9)
    nonhuman_90 = int(len(nonhuman_seqs) * 0.9)

    forget_seqs = [human_seqs[i] for i in human_idx[:human_90]]
    ambig_human = [human_seqs[i] for i in human_idx[human_90:]]

    adjacent_seqs = [nonhuman_seqs[i] for i in nonhuman_idx[:nonhuman_90]]
    ambig_nonhuman = [nonhuman_seqs[i] for i in nonhuman_idx[nonhuman_90:]]

    ambiguous_seqs = ambig_human + ambig_nonhuman

    # Deduplicate Swiss-Prot against all virus sequences, and filter viral proteins
    all_virus = set(human_seqs + nonhuman_seqs)
    retain_seqs, viral_swissprot = filter_swissprot(
        swissprot_seqs, all_virus, headers=swissprot_headers,
    )

    # Route Swiss-Prot viral proteins to the ambiguous set (default mode = all params update)
    ambiguous_seqs = ambiguous_seqs + viral_swissprot

    # Split each category into train/val/test
    # forget: ~90/5/5 of the 90% virus_human
    forget_train, forget_val, forget_test = _split_three(
        forget_seqs, (0.90, 0.05, 0.05), rng
    )
    # adjacent: same split
    adj_train, adj_val, adj_test = _split_three(
        adjacent_seqs, (0.90, 0.05, 0.05), rng
    )
    # retain: 95/2.5/2.5
    ret_train, ret_val, ret_test = _split_three(
        retain_seqs, (0.95, 0.025, 0.025), rng
    )

    print(f"\n--- Dataset sizes ---")
    print(f"Forget:    {len(forget_train)} train / {len(forget_val)} val / {len(forget_test)} test")
    print(f"Adjacent:  {len(adj_train)} train / {len(adj_val)} val / {len(adj_test)} test")
    print(f"Ambiguous: {len(ambiguous_seqs)} total (no split)")
    print(f"Retain:    {len(ret_train)} train / {len(ret_val)} val / {len(ret_test)} test")

    # Build and save datasets
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    splits_map = {
        "forget": (forget_train, forget_val, forget_test),
        "adjacent": (adj_train, adj_val, adj_test),
        "retain": (ret_train, ret_val, ret_test),
    }

    for name, (train, val, test) in splits_map.items():
        cat_dir = os.path.join(output_dir, name)
        for split_name, split_seqs in [("train", train), ("val", val), ("test", test)]:
            ds = _seqs_to_dataset(split_seqs)
            save_path = os.path.join(cat_dir, split_name)
            ds.save_to_disk(save_path)
            results[f"{name}_{split_name}"] = ds

    # Ambiguous: single dataset (no train/val/test split)
    ambig_ds = _seqs_to_dataset(ambiguous_seqs)
    ambig_dir = os.path.join(output_dir, "ambiguous")
    ambig_ds.save_to_disk(ambig_dir)
    results["ambiguous"] = ambig_ds

    print(f"\nDatasets saved to {output_dir}")
    return results


# ---------------------------------------------------------------------------
# MLM Collator
# ---------------------------------------------------------------------------

class MLMCollator:
    """Dynamic BERT-style MLM masking at batch time for ESM-2."""

    def __init__(self, alphabet, mask_ratio: float = 0.15, max_length: int = 1022):
        self.alphabet = alphabet
        self.mask_ratio = mask_ratio
        self.max_length = max_length

        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.pad_idx = alphabet.padding_idx

    def _tokenize(self, seq: str) -> torch.Tensor:
        """Tokenize a sequence: [cls] + AAs + [eos], truncated to max_length."""
        aa_indices = [self.alphabet.get_idx(aa) for aa in seq[: self.max_length]]
        tokens = [self.cls_idx] + aa_indices + [self.eos_idx]
        return torch.tensor(tokens, dtype=torch.long)

    def _apply_mask(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT masking: 80% mask, 10% random AA, 10% keep."""
        masked = tokens.clone()
        labels = torch.full_like(tokens, -100)

        special = {self.cls_idx, self.eos_idx, self.pad_idx}
        maskable = torch.tensor([t.item() not in special for t in tokens], dtype=torch.bool)
        n_maskable = maskable.sum().item()

        if n_maskable == 0:
            return masked, labels

        n_mask = max(1, int(n_maskable * self.mask_ratio))
        maskable_indices = torch.where(maskable)[0]
        chosen = maskable_indices[torch.randperm(len(maskable_indices))[:n_mask]]

        for idx in chosen:
            labels[idx] = tokens[idx]
            rand = torch.rand(1).item()
            if rand < 0.8:
                masked[idx] = self.mask_idx
            elif rand < 0.9:
                masked[idx] = torch.randint(4, 24, (1,)).item()
            # else: keep original

        return masked, labels

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_attention = []

        for item in batch:
            seq = item["sequence"]
            tokens = self._tokenize(seq)
            input_ids, labels = self._apply_mask(tokens)
            attention_mask = torch.ones(len(tokens), dtype=torch.long)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention.append(attention_mask)

        # Pad to max length in this batch
        max_len = max(t.size(0) for t in all_input_ids)

        padded_ids = []
        padded_labels = []
        padded_attn = []

        for ids, labs, attn in zip(all_input_ids, all_labels, all_attention):
            pad_len = max_len - ids.size(0)
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_idx, dtype=torch.long)]))
            padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
            padded_attn.append(torch.cat([attn, torch.zeros(pad_len, dtype=torch.long)]))

        return {
            "input_ids": torch.stack(padded_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attn),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SGTM data pipeline")
    parser.add_argument("--data-dir", default="data/sgtm", help="Output directory for processed datasets")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw virus TSV files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: Download Swiss-Prot
    fasta_path = os.path.join(args.raw_dir, "uniprot_sprot.fasta.gz")
    download_swissprot(fasta_path)

    # Step 2: Parse Swiss-Prot (with headers for viral protein detection)
    swissprot_seqs, swissprot_headers = parse_fasta_gz(fasta_path, return_headers=True)

    # Step 3: Load virus data
    human_seqs, nonhuman_seqs = load_virus_data(args.raw_dir)

    # Step 4: Prepare and save datasets
    prepare_datasets(
        human_seqs, nonhuman_seqs, swissprot_seqs, args.data_dir,
        seed=args.seed, swissprot_headers=swissprot_headers,
    )


if __name__ == "__main__":
    main()
