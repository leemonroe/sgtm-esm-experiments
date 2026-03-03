"""
Download viral protein data from UniProt for SGTM experiments.

Produces two TSV files:
  - virus_human.tsv: Reviewed viral proteins from species known to infect humans
  - virus_nonhuman.tsv: Reviewed viral proteins from species NOT known to infect humans

The human/non-human split uses UniProt's "virus_host" annotation. Proteins from
virus species that include Homo sapiens (TaxID: 9606) as a host are classified
as human-infecting.

NOTE on granularity: The host annotation is at the virus SPECIES level, not the
strain level. For example, all Influenza A proteins are tagged as human-infecting
even if the specific strain (e.g., A/Duck/England/1/1956 H11N6) is avian-only.
This is a known limitation of the data — see phase2_research_log.md.

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

# Columns to retrieve (match the original TSV format)
FIELDS = "accession,protein_name,organism_name,sequence,length,virus_hosts"

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

    # Step 3: Split by human host
    human, nonhuman = split_by_human_host(valid_rows)
    print(f"\nHuman-infecting (Virus hosts contains TaxID {HUMAN_TAXID}): {len(human)}")
    print(f"Non-human viral: {len(nonhuman)}")

    # Step 4: Write output files
    # Use column names matching the original TSV format
    fieldnames = ["Entry", "Protein names", "Organism", "Sequence", "Length", "Virus hosts"]

    # Map UniProt API field names to our TSV column names
    def remap_row(row):
        return {
            "Entry": row.get("Entry", ""),
            "Protein names": row.get("Protein names", ""),
            "Organism": row.get("Organism", ""),
            "Sequence": row.get("Sequence", ""),
            "Length": row.get("Length", ""),
            "Virus hosts": row.get("Virus hosts", ""),
        }

    human_path = os.path.join(args.output_dir, "virus_human.tsv")
    nonhuman_path = os.path.join(args.output_dir, "virus_nonhuman.tsv")

    write_tsv([remap_row(r) for r in human], human_path, fieldnames)
    write_tsv([remap_row(r) for r in nonhuman], nonhuman_path, fieldnames)

    print(f"\nWritten:")
    print(f"  {human_path} ({len(human)} sequences)")
    print(f"  {nonhuman_path} ({len(nonhuman)} sequences)")

    # Step 5: Sanity check against expected counts
    print(f"\n--- Sanity check ---")
    print(f"Expected ~1,239 human / ~349 non-human (from original data)")
    print(f"Got {len(human)} human / {len(nonhuman)} non-human")
    if abs(len(human) - 1239) > 100 or abs(len(nonhuman) - 349) > 100:
        print("WARNING: Counts differ significantly from original data.")
        print("UniProt is updated regularly — counts may change between releases.")
        print("Check if the query or filters need adjustment.")


if __name__ == "__main__":
    main()
