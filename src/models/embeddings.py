import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from utils.logging_utils import logger
from utils.config import *

_embedding_cache = {}  # (subset_id, embedding_name) -> np.ndarray

def embed_scibert(texts, model_name="allenai/scibert_scivocab_uncased", batch_size=16):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(DEVICE)
    mdl.eval()

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


def get_embeddings(texts, embedding_key, subset_id):
    """
    texts: list of strings
    embedding_key: one of EMBEDDING_MODELS keys
    subset_id: "cluster" or "class" (just for cache separation)
    Returns: L2-normalized embeddings (np.ndarray)
    """
    cache_key = (subset_id, embedding_key)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    logger.info("Computing embeddings: subset=%s, model=%s", subset_id, embedding_key)
    if embedding_key == "SciBERT":
        X = embed_scibert(texts)
    else:
        mdl = SentenceTransformer(EMBEDDING_MODELS[embedding_key])
        X = mdl.encode(texts, show_progress_bar=True).astype(np.float32)

    # normalize for k-means / GMM / HDBSCAN
    X_norm = normalize(X)
    _embedding_cache[cache_key] = X_norm
    return X_norm