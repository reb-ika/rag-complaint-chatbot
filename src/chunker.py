"""
chunker.py
Handles splitting complaint narratives into chunks for embedding.
"""

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_complaints(df, chunk_size=500, chunk_overlap=50):
    """
    Split complaint narratives into overlapping text chunks.

    Why chunking?
    Long narratives are ineffective when embedded as one vector.
    Smaller chunks allow more precise semantic search.

    Args:
        df (pd.DataFrame): Dataframe with cleaned_narrative column
        chunk_size (int): Max characters per chunk
        chunk_overlap (int): Characters to overlap between chunks

    Returns:
        pd.DataFrame: Dataframe with one row per chunk
    """
    # Validate input
    if 'cleaned_narrative' not in df.columns:
        raise ValueError("Dataframe must have a 'cleaned_narrative' column")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize text splitter: {e}")

    all_chunks = []

    for idx, row in df.iterrows():
        try:
            chunks = text_splitter.split_text(str(row['cleaned_narrative']))

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk_text': chunk,
                    'complaint_id': row.get('complaint_id', idx),
                    'product_category': row.get('product_category', 'Unknown'),
                    'issue': row.get('issue', ''),
                    'company': row.get('company', ''),
                    'state': row.get('state', ''),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })

        except Exception as e:
            print(f"Warning: skipping row {idx} due to error — {e}")
            continue

    df_chunks = pd.DataFrame(all_chunks)
    print(f"Created {len(df_chunks)} chunks from {len(df)} complaints")
    print(f"Average chunks per complaint: {len(df_chunks)/len(df):.1f}")
    return df_chunks