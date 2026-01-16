# 🧪 Research Article Clustering & Classification on arXiv

Unsupervised clustering + supervised transformer classification using scientific embeddings  
🔗 *Based on*: [Arcan (2025)](https://arxiv.org/abs/2601.08841)

---

## 📘 Overview

This project explores how to cluster scientific research abstracts without prior knowledge of category structure, and how to train a transformer model to classify documents based on discovered clusters.

The methodology integrates:

- State-of-the-art text embeddings (MiniLM, SciBERT, SPECTER)
- Clustering algorithms (KMeans, GMM, HDBSCAN)
- Transformer-based classifiers (BERT, DistilBERT, RoBERTa)
- Hyperparameter optimization using Optuna + experiment tracking via MLflow


![Abstract Length Distribution](analysis/distribution.png)


![Top Category Distribution](analysis/top_categories.png)

---

## 📊 Results

Empirical evaluation of unsupervised clustering (5,000 abstracts) and transformer-based classification (10,000 labeled abstracts using pseudo-labels). 

See [Arcan (2025)](https://arxiv.org/abs/2601.08841) for full methodology and benchmarks.

---

## ✅ Conclusion

This pipeline implements an approach to demonstrate that combining structured triples with unstructured scientific text improves both clustering coherence and classification accuracy. It provides a strong foundation for knowledge-aware semantic organization of research documents.

---

## 📎 BibTeX

```bibtex
@misc{arcan2025triplesknowledgeinfusedembeddingsclustering,
  title={Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents}, 
  author={Mihael Arcan},
  year={2025},
  eprint={2601.08841},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.08841}
}
```