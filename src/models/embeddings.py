import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from utils.config import (
    DEVICE,
    EMBEDDING_MODELS,
    TEXT_REPRESENTATION_CLUSTER,
    TEXT_REPRESENTATION_CLASS,
)


_embedding_cache = {}  # (subset_id, embedding_name) -> np.ndarray
_st_model_cache = {}   # model_name -> SentenceTransformer
_hf_model_cache = {}   # model_name -> (tokenizer, model)


def get_embeddings(df, embedding_key, subset_id, texts_override=None):

    # -------- Explicit override (cluster assignment) --------
    if texts_override is not None:
        return get_single_embedding(texts_override, embedding_key, subset_id)

    # -------- Select mode --------
    if subset_id == "cluster":
        mode = TEXT_REPRESENTATION_CLUSTER
    else:
        mode = TEXT_REPRESENTATION_CLASS

    # -------- Extract string fields --------
    texts_abs = df["clean"].tolist()

    def _format_triples(x):
        # CASE 1: your new extractor → string
        if isinstance(x, str):
            return x

        # CASE 2: old format → list of (s, r, o)
        if isinstance(x, list):
            out = []
            for item in x:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    s, r, o = item
                    out.append(f"{s} {r} {o}")
                else:
                    # fallback
                    out.append(str(item))
            return " ; ".join(out)

        # Unexpected type
        return str(x)

    texts_tri = df["triples"].apply(_format_triples).tolist()


    texts_abs_tri = df["abstract_triples"].tolist()

    texts_graph = df["graph"].tolist() if "graph" in df else None

    # -------------------------
    # 1) abstract
    # -------------------------
    if mode == "abstract":
        return get_single_embedding(texts_abs, embedding_key, subset_id+"_abs")

    # -------------------------
    # 2) triples
    # -------------------------
    if mode == "triples":
        return get_single_embedding(texts_tri, embedding_key, subset_id+"_tri")

    # -------------------------
    # 3) abstract_triples
    # -------------------------
    if mode == "abstract_triples":
        return get_single_embedding(texts_abs_tri, embedding_key, subset_id+"_abs_tri")

    # -------------------------
    # 4) hybrid = concatenate embeddings
    # -------------------------
    if mode == "hybrid":
        emb_abs = get_single_embedding(texts_abs, embedding_key, subset_id+"_abs")
        emb_tri = get_single_embedding(texts_tri, embedding_key, subset_id+"_tri")

        if texts_graph:
            emb_graph = get_single_embedding(texts_graph, embedding_key, subset_id+"_graph")
            X = np.concatenate([emb_abs, emb_tri, emb_graph], axis=1)
        else:
            X = np.concatenate([emb_abs, emb_tri], axis=1)

        return normalize(X)

    raise ValueError(f"Unknown TEXT_REPRESENTATION={mode}")


def get_single_embedding(texts, embedding_key, subset_id):
    cache_key = (subset_id, embedding_key)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    if embedding_key == "SciBERT":
        X = embed_scibert(texts)
    else:
        model_name = EMBEDDING_MODELS[embedding_key]
        mdl = _st_model_cache.get(model_name)
        if mdl is None:
            local_only = os.getenv("HF_HUB_OFFLINE", "0") == "1"
            mdl = SentenceTransformer(model_name, local_files_only=local_only)
            _st_model_cache[model_name] = mdl
        X = mdl.encode(texts, show_progress_bar=False).astype(np.float32)

    X_norm = normalize(X)
    _embedding_cache[cache_key] = X_norm
    return X_norm


def embed_scibert(texts, model_name="allenai/scibert_scivocab_uncased", batch_size=16):
    cached = _hf_model_cache.get(model_name)
    local_only = os.getenv("HF_HUB_OFFLINE", "0") == "1"
    if cached is None:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        mdl = AutoModel.from_pretrained(model_name, local_files_only=local_only).to(DEVICE)
        mdl.eval()
        _hf_model_cache[model_name] = (tok, mdl)
    else:
        tok, mdl = cached

    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            out = mdl(**enc)
            vec = out.last_hidden_state.mean(dim=1).cpu().numpy()
        outs.append(vec)
    emb = np.vstack(outs).astype(np.float32)
    return emb
