"""BERTopic latent theme discovery across the full corpus.

Runs BERTopic on 3,586 health practitioner disciplinary decisions using
MiniLM sentence-transformer embeddings, UMAP dimensionality reduction,
and HDBSCAN clustering.

Saves topic assignments, topic info, and visualisations.

Usage:
    python -m src.topic_modelling.bertopic_analysis
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
OUTPUT_PATH = DATA_DIR / "extracted" / "topic_assignments.jsonl"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODEL_DIR = PROJECT_ROOT / "models" / "bertopic"


def load_corpus() -> tuple[list[str], list[dict]]:
    """Load corpus texts and metadata."""
    texts = []
    metadata = []
    with open(CORPUS_PATH) as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["full_text"])
            metadata.append({
                "id": record["id"],
                "tribunal": record.get("tribunal", ""),
                "jurisdiction": record.get("jurisdiction", ""),
                "date": record.get("date", ""),
                "profession": record.get("profession", ""),
            })
    return texts, metadata


def truncate_texts(texts: list[str], max_chars: int = 10000) -> list[str]:
    """Truncate texts to a manageable length for embedding.

    Focus on findings and orders sections by taking the last portion,
    where the most distinctive content usually appears.
    """
    truncated = []
    for text in texts:
        if len(text) > max_chars:
            # Take first 2000 (intro/background) + last 8000 (findings/orders)
            truncated.append(text[:2000] + " ... " + text[-8000:])
        else:
            truncated.append(text)
    return truncated


def run_bertopic(texts: list[str]) -> tuple:
    """Run BERTopic on the corpus."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    print("Loading sentence-transformer model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Computing embeddings...")
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
    )

    print("Fitting BERTopic...")
    # Configure sub-models for this corpus size
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        nr_topics="auto",
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    return topic_model, topics, probs, embeddings


def save_results(
    topic_model, topics: list[int], metadata: list[dict],
) -> None:
    """Save topic assignments and topic info."""
    # Save topic assignments
    assignments = []
    for i, meta in enumerate(metadata):
        assignments.append({
            "id": meta["id"],
            "topic": topics[i],
        })

    with open(OUTPUT_PATH, "w") as f:
        for a in assignments:
            f.write(json.dumps(a) + "\n")
    print(f"Topic assignments saved to {OUTPUT_PATH}")

    # Save topic info as CSV
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(TABLES_DIR / "topic_info.csv", index=False)
    print(f"Topic info saved to {TABLES_DIR / 'topic_info.csv'}")


def save_visualisations(topic_model, topics, texts, embeddings) -> None:
    """Generate and save BERTopic visualisations."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Topic hierarchy
        fig = topic_model.visualize_hierarchy()
        fig.write_html(str(FIGURES_DIR / "topic_hierarchy.html"))
        print(f"  Saved topic_hierarchy.html")
    except Exception as e:
        print(f"  Skipped hierarchy: {e}")

    try:
        # Topic barchart (top 15 topics)
        fig = topic_model.visualize_barchart(top_n_topics=15)
        fig.write_html(str(FIGURES_DIR / "topic_barchart.html"))
        print(f"  Saved topic_barchart.html")
    except Exception as e:
        print(f"  Skipped barchart: {e}")

    try:
        # Intertopic distance map
        fig = topic_model.visualize_topics()
        fig.write_html(str(FIGURES_DIR / "topic_distance_map.html"))
        print(f"  Saved topic_distance_map.html")
    except Exception as e:
        print(f"  Skipped distance map: {e}")

    try:
        # Heatmap
        fig = topic_model.visualize_heatmap()
        fig.write_html(str(FIGURES_DIR / "topic_heatmap.html"))
        print(f"  Saved topic_heatmap.html")
    except Exception as e:
        print(f"  Skipped heatmap: {e}")


def print_topic_summary(topic_model, topics: list[int], metadata: list[dict]) -> None:
    """Print a summary of discovered topics."""
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # Exclude -1 (outlier topic)
    n_outliers = sum(1 for t in topics if t == -1)

    print(f"\n{'=' * 60}")
    print(f"TOPIC MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total documents: {len(topics)}")
    print(f"Topics discovered: {n_topics}")
    print(f"Outliers (topic -1): {n_outliers} ({n_outliers/len(topics)*100:.1f}%)")

    print(f"\nTop 15 topics:")
    for _, row in topic_info.head(16).iterrows():
        if row["Topic"] == -1:
            continue
        words = row.get("Representation", row.get("Name", ""))
        if isinstance(words, list):
            words = ", ".join(words[:5])
        print(f"  Topic {row['Topic']:>3} ({row['Count']:>4} docs): {words}")


def main():
    print("=== BERTopic Analysis ===\n")

    # Load corpus
    print("Loading corpus...")
    texts, metadata = load_corpus()
    print(f"  {len(texts)} decisions loaded")

    # Truncate for embedding efficiency
    print("Truncating texts for embedding...")
    texts_truncated = truncate_texts(texts)

    # Run BERTopic
    topic_model, topics, probs, embeddings = run_bertopic(texts_truncated)

    # Print summary
    print_topic_summary(topic_model, topics, metadata)

    # Save results
    print("\nSaving results...")
    save_results(topic_model, topics, metadata)

    # Save visualisations
    print("\nGenerating visualisations...")
    save_visualisations(topic_model, topics, texts_truncated, embeddings)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(MODEL_DIR / "bertopic_model"), serialization="safetensors", save_ctfidf=True, save_embedding_model="all-MiniLM-L6-v2")
    print(f"\nModel saved to {MODEL_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()
