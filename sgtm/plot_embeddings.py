"""
Visualize ESM-2 protein embeddings colored by biological category.

Extracts mean-pooled last-layer embeddings from pretrained ESM-2, then
projects to 2D (UMAP/t-SNE) and 3D (UMAP) to show how protein families
cluster in embedding space.

Usage:
  python -m sgtm.plot_embeddings --model-size 35M --data-dir data/sgtm/family_coronaviridae --device cuda
  python -m sgtm.plot_embeddings --model-size 8M --data-dir data/sgtm/family_coronaviridae --device cpu --max-seqs 200
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sgtm.model_config import load_alphabet

PRETRAINED_MODELS = {
    "8M": "esm2_t6_8M_UR50D",
    "35M": "esm2_t12_35M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
}

# Colors matching main plot palette
COLORS = {
    "Forget\n(Coronaviridae)": "#E24A33",
    "Adjacent\n(other viral)": "#FBC15E",
    "Retain\n(non-viral)": "#348ABD",
}


def load_pretrained(model_size, device):
    """Load pretrained ESM-2 from Meta."""
    import esm
    model_name = PRETRAINED_MODELS[model_size]
    load_fn = getattr(esm.pretrained, model_name)
    model, alphabet = load_fn()
    model = model.to(device)
    model.eval()
    return model, alphabet


def extract_embeddings(model, alphabet, sequences, device, batch_size=8):
    """Extract mean-pooled last-layer embeddings."""
    batch_converter = alphabet.get_batch_converter()
    embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_data = [(f"s{j}", seq[:1022]) for j, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers],
                            return_contacts=False)
            reps = results["representations"][model.num_layers]

            for j, seq in enumerate(batch_seqs):
                seq_len = min(len(seq), 1022)
                emb = reps[j, 1:seq_len + 1, :].mean(dim=0)
                embeddings.append(emb.cpu().numpy())

    return np.array(embeddings)


def load_split_sequences(data_dir, split_name, partition="val", max_seqs=None):
    """Load sequences from a data split."""
    from datasets import load_from_disk
    split_path = os.path.join(data_dir, split_name, partition)
    if not os.path.isdir(split_path):
        return []
    ds = load_from_disk(split_path)
    seqs = [row["sequence"] for row in ds]
    if max_seqs and len(seqs) > max_seqs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(seqs), size=max_seqs, replace=False)
        seqs = [seqs[i] for i in idx]
    return seqs


def plot_2d(embeddings_2d, labels, label_names, output_path):
    """2D scatter plot with biological category coloring."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F7F7F7",
        "font.size": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each category (retain first so forget is on top)
    for label_idx in reversed(range(len(label_names))):
        mask = labels == label_idx
        name = label_names[label_idx]
        color = COLORS.get(name, "#888888")
        n = mask.sum()
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=color, s=30, alpha=0.6, edgecolors="white",
                   linewidths=0.3, label=f"{name} (n={n})", zorder=label_idx + 2)

    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.set_title("ESM-2 Protein Embeddings by Biological Category\n"
                 "Coronaviridae clusters distinctly from other proteins",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11, framealpha=0.9, markerscale=2)

    # Remove tick labels (arbitrary projection axes)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d(embeddings_3d, labels, label_names, output_path):
    """3D scatter plot with biological category coloring."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for label_idx in reversed(range(len(label_names))):
        mask = labels == label_idx
        name = label_names[label_idx]
        color = COLORS.get(name, "#888888")
        n = mask.sum()
        ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                   c=color, s=20, alpha=0.5, edgecolors="white",
                   linewidths=0.2, label=f"{name} (n={n})")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title("ESM-2 Protein Embeddings (3D)\n"
                 "Coronaviridae forms a distinct cluster in embedding space",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, markerscale=2)

    # Clean up tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Nice viewing angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d_interactive(embeddings_3d, labels, label_names, output_path):
    """Interactive 3D scatter using plotly (saves as HTML)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping interactive 3D plot")
        return

    fig = go.Figure()

    color_map = {
        "Forget\n(Coronaviridae)": "#E24A33",
        "Adjacent\n(other viral)": "#FBC15E",
        "Retain\n(non-viral)": "#348ABD",
    }

    for label_idx in range(len(label_names)):
        mask = labels == label_idx
        name = label_names[label_idx]
        color = color_map.get(name, "#888888")
        display_name = name.replace("\n", " ")

        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers',
            name=f"{display_name} (n={mask.sum()})",
            marker=dict(size=3, color=color, opacity=0.6,
                        line=dict(width=0.5, color='white')),
        ))

    fig.update_layout(
        title="ESM-2 Protein Embeddings — Interactive 3D",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        width=900,
        height=700,
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    print(f"Saved interactive: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ESM-2 embeddings")
    parser.add_argument("--model-size", type=str, default="35M")
    parser.add_argument("--data-dir", type=str, default="data/sgtm/family_coronaviridae")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seqs", type=int, default=500,
                        help="Max sequences per split (for speed)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--no-3d", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    size_tag = args.model_size.lower()

    # Load sequences from each split
    print("Loading sequences...")
    splits = {
        "Forget\n(Coronaviridae)": load_split_sequences(
            args.data_dir, "forget", max_seqs=args.max_seqs),
        "Adjacent\n(other viral)": load_split_sequences(
            args.data_dir, "adjacent", max_seqs=args.max_seqs),
        "Retain\n(non-viral)": load_split_sequences(
            args.data_dir, "retain", max_seqs=args.max_seqs),
    }

    all_seqs = []
    all_labels = []
    label_names = []
    for i, (name, seqs) in enumerate(splits.items()):
        if seqs:
            print(f"  {name}: {len(seqs)} sequences")
            all_seqs.extend(seqs)
            all_labels.extend([i] * len(seqs))
            label_names.append(name)

    labels = np.array(all_labels)

    # Extract embeddings
    print(f"\nLoading pretrained ESM-2 {args.model_size}...")
    model, alphabet = load_pretrained(args.model_size, args.device)
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, alphabet, all_seqs, args.device, args.batch_size)
    print(f"  Embedding shape: {embeddings.shape}")

    # Free GPU memory
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Dimensionality reduction
    if args.method == "umap":
        from umap import UMAP
        print("\nRunning UMAP (2D)...")
        reducer_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          metric="cosine", random_state=42)
        emb_2d = reducer_2d.fit_transform(embeddings)

        if not args.no_3d:
            print("Running UMAP (3D)...")
            reducer_3d = UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                              metric="cosine", random_state=42)
            emb_3d = reducer_3d.fit_transform(embeddings)
    else:
        from sklearn.manifold import TSNE
        print("\nRunning t-SNE (2D)...")
        reducer_2d = TSNE(n_components=2, perplexity=30, random_state=42,
                          metric="cosine")
        emb_2d = reducer_2d.fit_transform(embeddings)

        if not args.no_3d:
            print("Running t-SNE (3D)...")
            reducer_3d = TSNE(n_components=3, perplexity=30, random_state=42,
                              metric="cosine")
            emb_3d = reducer_3d.fit_transform(embeddings)

    # Plot 2D
    plot_2d(emb_2d, labels, label_names,
            os.path.join(args.output_dir, f"embeddings_2d_{size_tag}.png"))

    # Plot 3D
    if not args.no_3d:
        plot_3d(emb_3d, labels, label_names,
                os.path.join(args.output_dir, f"embeddings_3d_{size_tag}.png"))
        plot_3d_interactive(emb_3d, labels, label_names,
                            os.path.join(args.output_dir, f"embeddings_3d_{size_tag}.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
