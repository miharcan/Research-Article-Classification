import torch

JSON_PATH = "/home/miharc/work/datasets/archive/arxiv-metadata-oai-snapshot.json"

LOAD_N_CLUSTERING = 5000
LOAD_N_CLASSIFIER = 10000

EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "SPECTER": "sentence-transformers/allenai-specter",
}

CLASSIFICATION_CANDIDATES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
]

CLUSTER_METHODS = ["kmeans", "gmm", "hdbscan"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
