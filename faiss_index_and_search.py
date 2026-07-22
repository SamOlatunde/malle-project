"""
Module: faiss_index_and_search.py

Builds a FAISS index from index embeddings and searches it with query embeddings.
All paths are derived from config.py — change knobs there, not here.
"""

import json
import os

import faiss

import config
from embed import load_embeddings, load_jsonl


os.makedirs(os.path.dirname(config.FAISS_INDEX), exist_ok=True)
os.makedirs(os.path.dirname(config.RESULTS), exist_ok=True)

# ── Build index ───────────────────────────────────────────────────────────────
index_embeddings = load_embeddings(config.EMBED_INDEX)
index_metadata   = load_jsonl(config.INDEX_META)

d     = index_embeddings.shape[1]
index = faiss.IndexFlatIP(d)   # inner product (vectors are L2-normalised)
index.add(index_embeddings)

faiss.write_index(index, config.FAISS_INDEX)

# ── Search ────────────────────────────────────────────────────────────────────
query_embeddings = load_embeddings(config.EMBED_QUERY)
query_metadata   = load_jsonl(config.QUERY_META)

k = 3
scores, indices = index.search(query_embeddings, k=k)

# ── Write results ─────────────────────────────────────────────────────────────
with open(config.RESULTS, 'w', encoding='utf-8') as f:
    for i, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        qinfo = query_metadata[i]
        matches = [
            {
                'score':             float(score),
                'index_id':          int(idx),
                'index_class':       index_metadata[idx]['class'],
                'index_instance_id': index_metadata[idx]['instance_id'],
                'index_path':        index_metadata[idx]['path'],
            }
            for score, idx in zip(row_scores, row_indices)
        ]
        entry = {
            'query_class':       qinfo['class'],
            'query_instance_id': qinfo['instance_id'],
            'query_path':        qinfo['path'],
            'matches':           matches,
        }
        f.write(json.dumps(entry) + '\n')
