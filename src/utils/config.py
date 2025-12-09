import torch
import spacy
import sys

meth_clus = sys.argv[1]
meth_clas = sys.argv[2]

JSON_PATH = "/home/miharc/work/datasets/archive/arxiv-metadata-oai-snapshot.json"

LOAD_N_CLUSTERING = 2000
LOAD_N_CLASSIFIER = 2000

n_train = 24 ##24

FORCE_K = None ###None     # e.g. FORCE_K = 8

EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "SPECTER": "sentence-transformers/allenai-specter",
}

CLASSIFICATION_CANDIDATES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "allenai/scibert_scivocab_uncased",
    "sentence-transformers/allenai-specter",
    "roberta-base",
]

CLUSTER_METHODS = ["kmeans", "gmm", "hdbscan"]

nlp = spacy.load("en_core_web_sm")

# Which text representation to use for clustering
TEXT_REPRESENTATION_CLUSTER = meth_clus ####"abstract", triples, abstract_triples, hybrid

# Which text representation to use for classification
TEXT_REPRESENTATION_CLASS = meth_clas ####"abstract", triples, abstract_triples, hybrid


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
