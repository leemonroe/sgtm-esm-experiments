"""
Download viral protein data from UniProt for SGTM experiments.

Produces TSV files split by viral taxonomy:
  - virus_human.tsv: Reviewed viral proteins from species known to infect humans
  - virus_nonhuman.tsv: Reviewed viral proteins from species NOT known to infect humans
  - virus_by_family.tsv: All reviewed viral proteins with family annotation

The human/non-human split uses UniProt's "virus_host" annotation at the SPECIES
level (not strain level) — see phase2_research_log.md for known limitations.

The family split uses UniProt's "lineage" field, extracting the "(family)" level
taxonomy. This enables family-based forget tasks (e.g., forget Coronaviridae).

Usage:
  python data/download_virus_data.py                          # default output to data/raw/
  python data/download_virus_data.py --output-dir data/raw/   # explicit output dir
  python data/download_virus_data.py --max-length 1022        # filter by length
"""

import argparse
import csv
import io
import os
import time
import urllib.parse
import urllib.request

# UniProt REST API (2022+ format)
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# Columns to retrieve
# lineage provides full taxonomy including family (e.g., "Coronaviridae (family)")
FIELDS = "accession,protein_name,organism_name,sequence,length,virus_hosts,lineage"

# Query for all reviewed viral proteins (Swiss-Prot, taxonomy: Viruses)
VIRUS_QUERY = "(taxonomy_id:10239) AND (reviewed:true)"

# Human host TaxID
HUMAN_TAXID = "9606"

# Sequence filters
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_uniprot_tsv(query, fields, batch_size=500):
    """Fetch results from UniProt REST API in TSV format, handling pagination."""
    all_rows = []
    headers = None
    cursor = None

    while True:
        params = {
            "query": query,
            "format": "tsv",
            "fields": fields,
            "size": str(batch_size),
        }
        if cursor:
            params["cursor"] = cursor

        url = f"{UNIPROT_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Python/sgtm-esm2-experiments")

        with urllib.request.urlopen(req) as response:
            # Check for next page cursor in Link header
            link_header = response.headers.get("Link", "")
            content = response.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(content), delimiter="\t")
        rows = list(reader)

        if headers is None and rows:
            headers = list(rows[0].keys())

        all_rows.extend(rows)
        print(f"  Fetched {len(all_rows)} entries...", end="\r")

        # Parse cursor from Link header for pagination
        if 'rel="next"' in link_header:
            # Extract cursor from URL in Link header
            import re
            match = re.search(r'cursor=([^&>]+)', link_header)
            if match:
                cursor = match.group(1)
            else:
                break
        else:
            break

        time.sleep(0.5)  # Be polite to the API

    print(f"  Fetched {len(all_rows)} entries total")
    return headers, all_rows


def is_valid_sequence(seq, min_length=30, max_length=1022):
    """Check standard AAs only and length within bounds."""
    return (
        min_length <= len(seq) <= max_length
        and all(aa in VALID_AAS for aa in seq)
    )


def extract_family(lineage_str):
    """Extract the viral family name from a UniProt lineage string.

    UniProt lineage fields look like:
      "Riboviria, Orthornavirae, ..., Coronaviridae, Betacoronavirus, ..."
    The family-level entry is tagged with "(family)" in some formats, or can
    be identified as the entry ending in "-viridae" or "-idae".

    Returns the family name or None if not found.
    """
    import re
    if not lineage_str:
        return None

    # First try: look for explicit "(family)" annotation
    match = re.search(r'(\w+)\s*\(family\)', lineage_str)
    if match:
        return match.group(1)

    # Fallback: find entries ending in -viridae (standard viral family suffix)
    for part in lineage_str.split(","):
        part = part.strip()
        if part.endswith("viridae"):
            return part

    return None


def split_by_human_host(rows):
    """Split viral proteins into human-infecting and non-human based on virus_hosts field."""
    human = []
    nonhuman = []

    for row in rows:
        hosts = row.get("Virus hosts", "")
        if HUMAN_TAXID in hosts:
            human.append(row)
        else:
            nonhuman.append(row)

    return human, nonhuman


def split_by_family(rows, target_family):
    """Split viral proteins by whether they belong to a specific viral family.

    Args:
        rows: List of row dicts with a "Lineage" field
        target_family: Family name to match (e.g., "Coronaviridae")

    Returns:
        (target_rows, other_rows)
    """
    target = []
    other = []

    for row in rows:
        lineage = row.get("Lineage", "")
        family = extract_family(lineage)
        if family and family.lower() == target_family.lower():
            target.append(row)
        else:
            other.append(row)

    return target, other


def write_tsv(rows, output_path, fieldnames):
    """Write rows to TSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Download viral protein data from UniProt")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Output directory for TSV files")
    parser.add_argument("--min-length", type=int, default=30,
                        help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=1022,
                        help="Maximum sequence length")
    parser.add_argument("--list-families", action="store_true",
                        help="Print family distribution and exit")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Fetch all reviewed viral proteins from UniProt
    print("Fetching reviewed viral proteins from UniProt...")
    print(f"  Query: {VIRUS_QUERY}")
    headers, rows = fetch_uniprot_tsv(VIRUS_QUERY, FIELDS)

    print(f"\nTotal reviewed viral proteins: {len(rows)}")

    # Step 2: Filter by sequence validity
    valid_rows = []
    for row in rows:
        seq = row.get("Sequence", "")
        if is_valid_sequence(seq, args.min_length, args.max_length):
            valid_rows.append(row)

    n_filtered = len(rows) - len(valid_rows)
    print(f"After filtering (length {args.min_length}-{args.max_length}, standard AAs): {len(valid_rows)} ({n_filtered} removed)")

    # Step 2.5: Extract family annotations
    from collections import Counter
    family_counts = Counter()
    no_family = 0
    for row in valid_rows:
        family = extract_family(row.get("Lineage", ""))
        if family:
            family_counts[family] += 1
        else:
            no_family += 1

    print(f"\nFamily annotation coverage: {sum(family_counts.values())}/{len(valid_rows)} "
          f"({no_family} without family)")

    if args.list_families:
        print(f"\n--- Top 30 viral families by sequence count ---")
        for family, count in family_counts.most_common(30):
            print(f"  {family}: {count}")
        return

    # Step 3: Split by human host
    human, nonhuman = split_by_human_host(valid_rows)
    print(f"\nHuman-infecting (Virus hosts contains TaxID {HUMAN_TAXID}): {len(human)}")
    print(f"Non-human viral: {len(nonhuman)}")

    # Step 4: Write output files
    fieldnames = ["Entry", "Protein names", "Organism", "Sequence", "Length",
                  "Virus hosts", "Lineage", "Family"]

    def remap_row(row):
        return {
            "Entry": row.get("Entry", ""),
            "Protein names": row.get("Protein names", ""),
            "Organism": row.get("Organism", ""),
            "Sequence": row.get("Sequence", ""),
            "Length": row.get("Length", ""),
            "Virus hosts": row.get("Virus hosts", ""),
            "Lineage": row.get("Lineage", ""),
            "Family": extract_family(row.get("Lineage", "")) or "",
        }

    # Write human/nonhuman splits (legacy format)
    human_path = os.path.join(args.output_dir, "virus_human.tsv")
    nonhuman_path = os.path.join(args.output_dir, "virus_nonhuman.tsv")
    write_tsv([remap_row(r) for r in human], human_path, fieldnames)
    write_tsv([remap_row(r) for r in nonhuman], nonhuman_path, fieldnames)

    # Write all viral proteins with family annotation
    all_viral_path = os.path.join(args.output_dir, "virus_by_family.tsv")
    write_tsv([remap_row(r) for r in valid_rows], all_viral_path, fieldnames)

    print(f"\nWritten:")
    print(f"  {human_path} ({len(human)} sequences)")
    print(f"  {nonhuman_path} ({len(nonhuman)} sequences)")
    print(f"  {all_viral_path} ({len(valid_rows)} sequences, with family annotation)")

    # Step 5: Print top families
    print(f"\n--- Top 10 viral families ---")
    for family, count in family_counts.most_common(10):
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
