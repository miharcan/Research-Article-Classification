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

Eklund et al. (2023) — An empirical configuration study of a common document clustering pipeline.  (1)
- Evaluated BERT, Doc2Vec, UMAP, PCA, KMeans, HDBSCAN.
- Found that embedding quality dominates clustering performance.
- Showed that both KMeans and HDBSCAN work well when given strong embeddings.

Relevance: Our pipeline compares MiniLM, SciBERT, and SPECTER embeddings across multiple clustering algorithms.

Asyaky & Mandala (2021) — Improving the Performance of HDBSCAN on Short Text Clustering by Using Word Embedding and UMAP (2)
- HDBSCAN handles arbitrary cluster shapes and density variations better than K-means.
- Automatically determines number of clusters.
- Leaves noise points unassigned → avoids forcing ambiguous documents into clusters.

Relevance: Demonstrates including HDBSCAN in the clustering comparison and evaluating noise ratios.


Wolff et al. (2024) — Enriched BERT Embeddings for Scholarly Publication Classification (3)
- Compared BERT, SciBERT, SciNCL, SPECTER2 for arXiv-like document classification.
- Domain-specialized transformers (SciBERT, SPECTER2) outperform vanilla BERT.
- Best models achieve weighted F1 ≈ 0.74 across 123 categories.

Relevance: Validates using transformer fine-tuning on scientific text.

Cohan et al. (2020) — SPECTER: Document-level Representation Learning using Citation-informed Transformers (4)
- Learned embeddings using citation graph supervision → superior topic separation.
- Achieved F1 = 86.4% on scientific classification benchmarks.
- Citation-aware embeddings capture semantic relatedness between papers.

Relevance: Supports inclusion of SciBERT/SPECTER embeddings in the clustering step.


Methodology
1. Exploratory Data Analysis, performed full scan of arXiv metadata:
- Abstract length statistics
- Distribution of top-level categories
- Identification of missing fields
- Histograms and top-N barplots saved in analysis/

2. Embedding models used:
- MiniLM-L6-v2 (fast, strong baseline)
- MPNet
- SciBERT
- SPECTER
All embeddings are: Batch encoded + L2-normalized + Cached per dataset split

3. Clustering Pipeline
- KMeans
- Gaussian Mixture Model (GMM)
- HDBSCAN

Metrics used:
- ARI (Adjusted Rand Index)
- NMI (Normalized Mutual Information)
- Silhouette Score
- Noise Ratio (for HDBSCAN)

The best clustering pipeline is selected by a composite score:
score = NMI + 0.5 * ARI + 0.5 * silhouette

4. Classification Pipeline
- Transformer model candidates: BERT, DistilBERT, RoBERTa
- Labels = cluster IDs (pseudo-labels)
- Framework: PyTorch
- Search: Optuna
- Tracking & visual artifacts: MLflow

Metrics include:
- Accuracy
- F1 macro / weighted
- Precision / Recall
- Cohen’s Kappa
- Matthews Correlation Coefficient
- Top-3 Accuracy
- ROC-AUC (multiclass OVR)


Results:
Below are the real results from running the full pipeline (2,000 abstracts total).

1. Clustering Results
Best number of clusters: K = 6
Best pipeline:
- Embedding: MiniLM  
- Algorithm: KMeans  
- Score: 0.8618

2. Classification Results (Optuna Search)

Best model discovered:
- distilbert-base-uncased
- lr = 4.62e-05
- batch = 16
- epochs = 4

Accuracy: 0.82
F1 Macro: 0.814
F1 Weighted: 0.819
Kappa: 0.780
MCC: 0.781
Top-3 Accuracy: 0.97
ROC-AUC (macro OVR): 0.967


How to Run
1. Install dependencies
pip install -r requirements.txt

2. Provide the arXiv dataset

Download from Kaggle:
https://www.kaggle.com/datasets/Cornell-University/arxiv

Then set the path in utils/config.py: JSON_PATH = "/path/to/arxiv-metadata-oai-snapshot.json"

3. Run the full pipeline
python src/main.py


Output:
- Logs → logs/
- EDA → analysis/
- Optuna search results → optuna_trial_results.csv
- MLflow artifacts → logged automatically

Scalability & Practical Considerations:
- Embeddings and clustering scale linearly with dataset size.
- HDBSCAN is more expensive; KMeans remains the fastest option.
- DistilBERT provides a strong accuracy-speed balance.
- The modular pipeline allows substituting any component (embeddings, clustering, classifier).

Limitations:
- Classification relies on pseudo-labels (cluster IDs), not true arXiv subjects.
- Stronger embeddings (e.g., SPECTER2) could further improve results.
- Dimensionality reduction (UMAP) is not yet integrated.


Conclusion: 
This is systematic approach to discovering latent scientific categories and training transformer models for classification. By combining:
- Embedding-based clustering
- Rigorous evaluation
- Hyperparameter optimization
…we obtain meaningful scientific clusters and a high-performing classifier.


References:
(1) Eklund, A., Forsman, M., &#38; Drewes, F. (2023). An empirical configuration study of a common document clustering pipeline. Northern European Journal of Language Technology (NEJLT)

(2) M. S. Asyaky and R. Mandala, "Improving the Performance of HDBSCAN on Short Text Clustering by Using Word Embedding and UMAP," 2021 8th International Conference on Advanced Informatics: Concepts, Theory and Applications (ICAICTA), Bandung, Indonesia, 2021, pp. 1-6, doi: 10.1109/ICAICTA53211.2021.9640285

(3) Wolff, B., Seidlmayer, E., Förstner, K.U. (2024). Enriched BERT Embeddings for Scholarly Publication Classification. In: Rehm, G., Dietze, S., Schimmler, S., Krüger, F. (eds) Natural Scientific Language Processing and Research Knowledge Graphs. NSLP 2024. Lecture Notes in Computer Science(), vol 14770. Springer, Cham. https://doi.org/10.1007/978-3-031-65794-8_16

(4) Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel Weld. 2020. SPECTER: Document-level Representation Learning using Citation-informed Transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2270–2282, Online. Association for Computational Linguistics.