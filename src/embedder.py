"""
embedder.py
Handles generating embeddings and building the FAISS vector store.
"""

import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer


def load_model(model_name='all-MiniLM-L6-v2'):
    """
    Load the sentence transformer embedding model.

    Args:
        model_name (str): HuggingFace model name

    Returns:
        SentenceTransformer: Loaded model
    """
    try:
        print(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def generate_embeddings(model, texts, batch_size=64):
    """
    Generate embeddings for a list of text chunks.

    Args:
        model: SentenceTransformer model
        texts (list): List of text strings to embed
        batch_size (int): Number of texts to process at once

    Returns:
        np.ndarray: Embedding matrix of shape (n_texts, embedding_dim)
    """
    if not texts:
        raise ValueError("texts list is empty")

    try:
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {e}")


def build_faiss_index(embeddings):
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings (np.ndarray): Embedding matrix

    Returns:
        faiss.Index: Built FAISS index
    """
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        print(f"FAISS index built with {index.ntotal} vectors")
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS index: {e}")


def save_vector_store(index, embeddings, df_chunks, vector_store_path):
    """
    Save the FAISS index, embeddings, and metadata to disk.

    Args:
        index: FAISS index
        embeddings (np.ndarray): Embedding matrix
        df_chunks (pd.DataFrame): Chunk metadata
        vector_store_path (str): Directory to save files
    """
    try:
        os.makedirs(vector_store_path, exist_ok=True)

        faiss.write_index(index, os.path.join(vector_store_path, 'complaints.index'))
        np.save(os.path.join(vector_store_path, 'embeddings.npy'), embeddings)
        df_chunks.to_csv(os.path.join(vector_store_path, 'chunks_metadata.csv'), index=False)

        print(f"Vector store saved to {vector_store_path}/")
        print(f"  - complaints.index")
        print(f"  - embeddings.npy")
        print(f"  - chunks_metadata.csv")
    except Exception as e:
        raise RuntimeError(f"Failed to save vector store: {e}")