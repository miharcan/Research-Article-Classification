# Research Article Clustering & Classification on arXiv
Unsupervised clustering + supervised transformer classification using scientific embeddings

Overview

This project explores how to cluster scientific research abstracts without knowing the number of categories, and how to train a transformer model to classify documents based on those discovered clusters.

The methodology integrates:
- State-of-the-art text embeddings (MiniLM, SciBERT, SPECTER)
- Clustering algorithms (KMeans, GMM, HDBSCAN)
- Transformer-based classifiers (BERT, DistilBERT, RoBERTa)
- Hyperparameter search using Optuna + MLflow

Literature Review

Eklund et al. (2023) — Benchmarking Modern Document Clustering Pipelines
- Evaluated BERT, Doc2Vec, UMAP, PCA, KMeans, HDBSCAN.
- Found that embedding quality dominates clustering performance.
- Showed that both KMeans and HDBSCAN work well when given strong embeddings.
- Supports this project’s modular pipeline:
- Embeddings → Clustering → Evaluation

Relevance: Our pipeline compares MiniLM, SciBERT, and SPECTER embeddings across multiple clustering algorithms.

Asyaky & Mandala (2021) — HDBSCAN for Short Texts
- HDBSCAN handles arbitrary cluster shapes and density variations better than K-means.
- Automatically determines number of clusters.
- Leaves noise points unassigned → avoids forcing ambiguous documents into clusters.
- Relevance: This justifies including HDBSCAN in the clustering comparison and evaluating noise ratios.
- K-Means vs. GMM (Classical Clustering Theory)
- K-means is a special case of Gaussian Mixture Models with equal spherical covariance.
- GMM supports elliptical clusters and soft assignments.
- In text embedding space, cluster shapes may be elongated or overlapping.

Relevance: Motivates inclusion of GMM to complement K-Means.

Wolff et al. (2024) — Field-of-Research Classification with Transformers
- Compared BERT, SciBERT, SciNCL, SPECTER2 for arXiv-like document classification.
- Domain-specialized transformers (SciBERT, SPECTER2) outperform vanilla BERT.
- Best models achieve weighted F1 ≈ 0.74 across 123 categories.

Relevance: Validates using transformer fine-tuning on scientific text.

Cohan et al. (2020) — SPECTER Embeddings
- Learned embeddings using citation graph supervision → superior topic separation.
- Achieved F1 = 86.4% on scientific classification benchmarks.
- Citation-aware embeddings capture semantic relatedness between papers.

Relevance: Supports inclusion of SciBERT/SPECTER embeddings in the clustering step.


🧪 Methodology
1. Exploratory Data Analysis
Performed full scan of arXiv metadata:
Abstract length statistics
Distribution of top-level categories
Identification of missing fields
Histograms and top-N barplots saved in analysis/

2. Embedding Generation
Embedding models used:
MiniLM-L6-v2 (fast, strong baseline)
MPNet
SciBERT
SPECTER
All embeddings are:
Batch encoded
L2-normalized
Cached per dataset split

3. Clustering Pipeline
Algorithms tested:
KMeans
Gaussian Mixture Model (GMM)
HDBSCAN

Metrics used:
ARI (Adjusted Rand Index)
NMI (Normalized Mutual Information)
Silhouette Score
Noise Ratio (for HDBSCAN)


The best clustering pipeline is selected by a composite score:
score = NMI + 0.5 * ARI + 0.5 * silhouette

4. Classification Pipeline

Transformer model candidates:
BERT, DistilBERT, RoBERTa
Labels = cluster IDs (pseudo-labels)
Framework: PyTorch
Search: Optuna
Tracking & visual artifacts: MLflow

Metrics include:
Accuracy
F1 macro / weighted
Precision / Recall
Cohen’s Kappa
Matthews Correlation Coefficient
Top-3 Accuracy
ROC-AUC (multiclass OVR)


📊 Results
Below are the real results from running the full pipeline (2,000 abstracts total).
1. Clustering Results
Best number of clusters: K = 6

Best pipeline:
Embedding: MiniLM  
Algorithm: KMeans  
Score: 0.8618

Cluster Interpretability
Cluster	Dominant Field
0	Condensed Matter Physics
1	Mixed Math / CS / Physics
2	High-Energy Physics (hep-ph / nucl-th)
3	Pure Mathematics
4	Astrophysics
5	Theoretical Physics (hep-th, gr-qc)

This aligns extremely well with known arXiv category structure.

2. Classification Results (Optuna Search)

Best model discovered:
distilbert-base-uncased
lr = 4.62e-05
batch = 16
epochs = 4

Best accuracy: 0.82
Metric	Result
Accuracy	0.82
F1 Macro	0.814
F1 Weighted	0.819
Kappa	0.780
MCC	0.781
Top-3 Accuracy	0.97
ROC-AUC (macro OVR)	0.967

Even without true labels (clusters used as pseudo-labels), the classifier achieves high consistency with the cluster structure.

🚀 How to Run
1. Install dependencies
pip install -r requirements.txt

2. Provide the arXiv dataset

Download from Kaggle:
https://www.kaggle.com/datasets/Cornell-University/arxiv

Then set the path in utils/config.py:

JSON_PATH = "/path/to/arxiv-metadata-oai-snapshot.json"

3. Run the full pipeline
python src/main.py


Output:
Logs → logs/
EDA → analysis/
Optuna search results → optuna_trial_results.csv
MLflow artifacts → logged automatically

Scalability & Practical Considerations
Embeddings and clustering scale linearly with dataset size.
HDBSCAN is more expensive; KMeans remains the fastest option.
DistilBERT provides a strong accuracy-speed balance.
The modular pipeline allows substituting any component (embeddings, clustering, classifier).

Limitations
Classification relies on pseudo-labels (cluster IDs), not true arXiv subjects.
Stronger embeddings (e.g., SPECTER2) could further improve results.
Dimensionality reduction (UMAP) is not yet integrated.


Conclusion

This project demonstrates a research-driven, systematic approach to discovering latent scientific categories and training transformer models for classification. By combining:
- Embedding-based clustering
- Rigorous evaluation
- Hyperparameter optimization
…we obtain meaningful scientific clusters and a high-performing classifier.
