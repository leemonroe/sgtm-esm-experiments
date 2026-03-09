"""
SGTM data pipeline: download Swiss-Prot, split virus/non-virus data into
forget/adjacent/retain categories, and save as HuggingFace datasets.

Supports multiple forget task granularities:
  - "coarse":  forget = ALL viral proteins, retain = non-viral (no adjacent)
  - "fine":    forget = human-infecting viral, adjacent = other viral, retain = non-viral
  - "family":  forget = one viral family (e.g. Coronaviridae), adjacent = other viral,
               retain = non-viral. Uses UniProt lineage taxonomy.
  - "custom":  forget = user-provided TSV, adjacent = remaining viral, retain = non-viral

Usage:
  python -m sgtm.data_pipeline --forget-task coarse
  python -m sgtm.data_pipeline --forget-task fine
  python -m sgtm.data_pipeline --forget-task family --forget-family Coronaviridae
  python -m sgtm.data_pipeline --forget-task custom --forget-tsv data/raw/my_forget_set.tsv
"""

import argparse
import gzip
import os
import re
import urllib.request
from typing import Dict, List, Optional, Tuple

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
    """Parse a gzipped FASTA file."""
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


def load_tsv_sequences(path: str) -> List[str]:
    """Load sequences from a TSV file with a 'Sequence' column (index 3)."""
    seqs = []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                seqs.append(parts[3])
    return seqs


def load_tsv_with_family(path: str) -> List[dict]:
    """Load sequences from virus_by_family.tsv with family annotation.

    Returns list of dicts with 'sequence' and 'family' keys.
    """
    import csv
    entries = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entries.append({
                "sequence": row.get("Sequence", ""),
                "family": row.get("Family", ""),
            })
    return entries


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
    """Check if a Swiss-Prot FASTA header indicates a viral organism."""
    match = _OS_PATTERN.search(header)
    if not match:
        return False
    organism = match.group(1).lower()
    return any(kw in organism for kw in _VIRAL_KEYWORDS)


def filter_swissprot(
    swissprot_seqs: List[str],
    seqs_to_exclude: set,
    headers: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """Filter Swiss-Prot into non-viral (retain) and viral sequences.

    Returns:
        (retain_seqs, viral_swissprot_seqs)
    """
    retain = []
    viral_swissprot = []
    seen = set()

    for i, seq in enumerate(swissprot_seqs):
        if not is_valid_sequence(seq):
            continue
        if seq in seqs_to_exclude:
            continue
        if seq in seen:
            continue
        seen.add(seq)

        if headers and _is_viral_header(headers[i]):
            viral_swissprot.append(seq)
        else:
            retain.append(seq)

    print(f"Swiss-Prot after filtering: {len(retain) + len(viral_swissprot):,} / {len(swissprot_seqs):,}")
    print(f"  Viral (by header): {len(viral_swissprot):,}")
    print(f"  Non-viral (retain): {len(retain):,}")
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


def _save_category(seqs_train, seqs_val, seqs_test, name, output_dir, results):
    """Save a train/val/test split as HuggingFace datasets."""
    cat_dir = os.path.join(output_dir, name)
    for split_name, split_seqs in [("train", seqs_train), ("val", seqs_val), ("test", seqs_test)]:
        ds = _seqs_to_dataset(split_seqs)
        ds.save_to_disk(os.path.join(cat_dir, split_name))
        results[f"{name}_{split_name}"] = ds
    print(f"  {name}: {len(seqs_train)} train / {len(seqs_val)} val / {len(seqs_test)} test")


# ---------------------------------------------------------------------------
# Dataset preparation (task-specific)
# ---------------------------------------------------------------------------

def prepare_coarse(
    all_viral_seqs: List[str],
    retain_seqs: List[str],
    output_dir: str,
    rng,
) -> Dict[str, Dataset]:
    """Coarse task: forget ALL viral proteins, retain non-viral.

    No adjacent category — clean two-way split.
    """
    print(f"\n--- Coarse task: viral vs non-viral ---")
    print(f"Forget (all viral): {len(all_viral_seqs)}")
    print(f"Retain (non-viral): {len(retain_seqs)}")

    forget_train, forget_val, forget_test = _split_three(
        all_viral_seqs, (0.90, 0.05, 0.05), rng
    )
    ret_train, ret_val, ret_test = _split_three(
        retain_seqs, (0.95, 0.025, 0.025), rng
    )

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print(f"\n--- Dataset sizes ---")
    _save_category(forget_train, forget_val, forget_test, "forget", output_dir, results)
    _save_category(ret_train, ret_val, ret_test, "retain", output_dir, results)

    # Write manifest
    manifest = {
        "forget_task": "coarse",
        "categories": ["forget", "retain"],
        "forget_description": "All viral proteins (curated + Swiss-Prot header-detected)",
        "retain_description": "Non-viral Swiss-Prot proteins",
    }
    _write_manifest(manifest, output_dir)

    return results


def prepare_fine(
    forget_seqs: List[str],
    adjacent_seqs: List[str],
    retain_seqs: List[str],
    output_dir: str,
    rng,
    forget_description: str = "Human-infecting viral proteins",
    adjacent_description: str = "Other viral proteins (non-human + Swiss-Prot viral)",
) -> Dict[str, Dataset]:
    """Fine task: forget a subset of viral proteins, adjacent = remaining viral.

    Three-way split matching the original SGTM paper's design.
    """
    print(f"\n--- Fine task ---")
    print(f"Forget: {len(forget_seqs)} ({forget_description})")
    print(f"Adjacent: {len(adjacent_seqs)} ({adjacent_description})")
    print(f"Retain: {len(retain_seqs)}")

    forget_train, forget_val, forget_test = _split_three(
        forget_seqs, (0.90, 0.05, 0.05), rng
    )
    adj_train, adj_val, adj_test = _split_three(
        adjacent_seqs, (0.90, 0.05, 0.05), rng
    )
    ret_train, ret_val, ret_test = _split_three(
        retain_seqs, (0.95, 0.025, 0.025), rng
    )

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print(f"\n--- Dataset sizes ---")
    _save_category(forget_train, forget_val, forget_test, "forget", output_dir, results)
    _save_category(adj_train, adj_val, adj_test, "adjacent", output_dir, results)
    _save_category(ret_train, ret_val, ret_test, "retain", output_dir, results)

    manifest = {
        "forget_task": "fine",
        "categories": ["forget", "adjacent", "retain"],
        "forget_description": forget_description,
        "adjacent_description": adjacent_description,
        "retain_description": "Non-viral Swiss-Prot proteins",
    }
    _write_manifest(manifest, output_dir)

    return results


def _write_manifest(manifest: dict, output_dir: str):
    """Write a JSON manifest describing the dataset configuration."""
    import json
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {path}")


# ---------------------------------------------------------------------------
# MLM Collator
# ---------------------------------------------------------------------------

class MLMCollator:
    """Dynamic BERT-style MLM masking at batch time for ESM-2.

    Args:
        seed: If provided, creates a dedicated torch.Generator so masking is
              deterministic and isolated from the global RNG. This is important
              for evaluation: pre- and post-ablation PPL must use the same masks
              regardless of other operations that may consume global RNG state.
              Leave None for training (stochastic masking is desirable).
    """

    def __init__(self, alphabet, mask_ratio: float = 0.15, max_length: int = 1022,
                 seed: int = None):
        self.alphabet = alphabet
        self.mask_ratio = mask_ratio
        self.max_length = max_length

        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.pad_idx = alphabet.padding_idx

        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)
        else:
            self.rng = None

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
        perm = torch.randperm(len(maskable_indices), generator=self.rng)
        chosen = maskable_indices[perm[:n_mask]]

        for idx in chosen:
            labels[idx] = tokens[idx]
            rand = torch.rand(1, generator=self.rng).item()
            if rand < 0.8:
                masked[idx] = self.mask_idx
            elif rand < 0.9:
                masked[idx] = torch.randint(4, 24, (1,), generator=self.rng).item()
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
    parser.add_argument("--forget-task", choices=["coarse", "fine", "family", "custom"],
                        default="coarse",
                        help="Forget task granularity: "
                             "coarse = all viral vs non-viral; "
                             "fine = human-infecting vs other; "
                             "family = one viral family vs other viral; "
                             "custom = user-provided forget TSV")
    parser.add_argument("--forget-family", type=str, default=None,
                        help="Viral family to forget (for --forget-task family, "
                             "e.g. Coronaviridae)")
    parser.add_argument("--forget-tsv", type=str, default=None,
                        help="Path to custom forget set TSV (for --forget-task custom)")
    parser.add_argument("--data-dir", default="data/sgtm",
                        help="Output directory for processed datasets")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory containing raw virus TSV files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)

    # Step 1: Download Swiss-Prot
    fasta_path = os.path.join(args.raw_dir, "uniprot_sprot.fasta.gz")
    download_swissprot(fasta_path)

    # Step 2: Parse Swiss-Prot
    swissprot_seqs, swissprot_headers = parse_fasta_gz(fasta_path, return_headers=True)

    # Step 3: Load curated virus data
    human_path = os.path.join(args.raw_dir, "virus_human.tsv")
    nonhuman_path = os.path.join(args.raw_dir, "virus_nonhuman.tsv")

    human_seqs = load_tsv_sequences(human_path) if os.path.exists(human_path) else []
    nonhuman_seqs = load_tsv_sequences(nonhuman_path) if os.path.exists(nonhuman_path) else []

    human_seqs = [s for s in human_seqs if is_valid_sequence(s)]
    nonhuman_seqs = [s for s in nonhuman_seqs if is_valid_sequence(s)]
    print(f"Curated virus data: {len(human_seqs)} human, {len(nonhuman_seqs)} non-human")

    # Step 4: Filter Swiss-Prot
    all_curated_virus = set(human_seqs + nonhuman_seqs)
    retain_seqs, viral_swissprot = filter_swissprot(
        swissprot_seqs, all_curated_virus, headers=swissprot_headers,
    )

    # Step 5: Build datasets based on forget task
    # Append output dir with task name for clarity
    output_dir = os.path.join(args.data_dir, args.forget_task)

    if args.forget_task == "coarse":
        # Forget = all viral (curated + Swiss-Prot detected)
        all_viral = human_seqs + nonhuman_seqs + viral_swissprot
        prepare_coarse(all_viral, retain_seqs, output_dir, rng)

    elif args.forget_task == "fine":
        # Forget = human-infecting, adjacent = non-human curated + Swiss-Prot viral
        adjacent_seqs = nonhuman_seqs + viral_swissprot
        prepare_fine(
            human_seqs, adjacent_seqs, retain_seqs, output_dir, rng,
            forget_description="Human-infecting viral proteins (species-level annotation)",
            adjacent_description="Non-human viral (curated) + Swiss-Prot viral (header-detected)",
        )

    elif args.forget_task == "family":
        if not args.forget_family:
            parser.error("--forget-family required for --forget-task family")

        # Load the family-annotated TSV
        family_tsv = os.path.join(args.raw_dir, "virus_by_family.tsv")
        if not os.path.exists(family_tsv):
            print(f"ERROR: {family_tsv} not found.")
            print("Run: python data/download_virus_data.py --output-dir data/raw/")
            return

        entries = load_tsv_with_family(family_tsv)
        entries = [e for e in entries if is_valid_sequence(e["sequence"])]

        # Split by target family
        forget_seqs = [e["sequence"] for e in entries
                       if e["family"].lower() == args.forget_family.lower()]
        adjacent_viral = [e["sequence"] for e in entries
                          if e["family"].lower() != args.forget_family.lower()]

        if not forget_seqs:
            print(f"ERROR: No sequences found for family '{args.forget_family}'")
            print("Run: python data/download_virus_data.py --list-families")
            return

        # Deduplicate against Swiss-Prot viral set already found
        forget_set = set(forget_seqs)
        # Adjacent = other viral from family TSV + Swiss-Prot header-detected viral
        # (deduplicated, excluding forget sequences)
        adj_set = set(adjacent_viral)
        for s in viral_swissprot:
            if s not in forget_set:
                adj_set.add(s)
        adjacent_seqs = list(adj_set)

        # Also remove forget sequences from retain (in case of overlap)
        retain_seqs_clean = [s for s in retain_seqs if s not in forget_set]

        output_dir = os.path.join(args.data_dir, f"family_{args.forget_family.lower()}")
        prepare_fine(
            forget_seqs, adjacent_seqs, retain_seqs_clean, output_dir, rng,
            forget_description=f"{args.forget_family} proteins (taxonomy-based)",
            adjacent_description=f"Other viral proteins (non-{args.forget_family})",
        )

    elif args.forget_task == "custom":
        if not args.forget_tsv:
            parser.error("--forget-tsv required for --forget-task custom")
        custom_seqs = [s for s in load_tsv_sequences(args.forget_tsv) if is_valid_sequence(s)]
        custom_set = set(custom_seqs)

        # Adjacent = all viral NOT in the custom forget set
        adjacent_seqs = [s for s in (human_seqs + nonhuman_seqs + viral_swissprot)
                         if s not in custom_set]
        prepare_fine(
            custom_seqs, adjacent_seqs, retain_seqs, output_dir, rng,
            forget_description=f"Custom forget set from {args.forget_tsv}",
            adjacent_description="Remaining viral proteins not in forget set",
        )

    print(f"\nDone. Data saved to {output_dir}")
    print(f"Use --data-dir {output_dir} when running training/eval scripts.")


if __name__ == "__main__":
    main()
