"""
retriever.py
Handles semantic search over the FAISS vector store.
"""

import numpy as np
import faiss
import pandas as pd
import os


def load_vector_store(vector_store_path):
    """
    Load the FAISS index and chunk metadata from disk.

    Args:
        vector_store_path (str): Path to vector_store directory

    Returns:
        tuple: (faiss.Index, pd.DataFrame)
    """
    index_path = os.path.join(vector_store_path, 'complaints.index')
    metadata_path = os.path.join(vector_store_path, 'chunks_metadata.csv')

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    try:
        index = faiss.read_index(index_path)
        df_chunks = pd.read_csv(metadata_path)
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        print(f"Loaded metadata with {len(df_chunks)} chunks")
        return index, df_chunks
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")


def retrieve(query, model, index, df_chunks, k=5, product_filter=None):
    """
    Retrieve the top-k most relevant complaint chunks for a query.

    How it works:
    1. Embed the query using the same model used for chunks
    2. Search the FAISS index for the k nearest vectors
    3. Return the matching chunk texts and metadata

    Args:
        query (str): User's question
        model: SentenceTransformer model
        index: FAISS index
        df_chunks (pd.DataFrame): Chunk metadata
        k (int): Number of chunks to retrieve
        product_filter (str): Optional product category to filter by

    Returns:
        list: List of dicts with chunk_text and metadata
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        # Embed the query
        query_vector = model.encode([query], convert_to_numpy=True).astype('float32')

        # Search FAISS index
        distances, indices = index.search(query_vector, k * 2)

        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            chunk = df_chunks.iloc[idx].to_dict()
            chunk['similarity_score'] = float(1 / (1 + dist))

            # Apply optional product filter
            if product_filter:
                if chunk.get('product_category') != product_filter:
                    continue

            results.append(chunk)

            if len(results) >= k:
                break

        return results

    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}")