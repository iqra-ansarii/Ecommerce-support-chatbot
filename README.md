# E-commerce Customer Support Chatbot

## Project Overview

This project implements an e-commerce customer-support chatbot using RAG,  Machine Learning, and semantic search.
The chatbot takes a customer's question and returns the most relevant support response from the dataset.

## Dataset

The project uses the Bitext Retail E-commerce Chatbot Training Dataset from Hugging Face.
- 44,884 records
- 46 intents
- 13 categories
- 5 columns: instruction, intent, category, tags, and response

## Models

### TF-IDF + Logistic Regression

The baseline intent-classification model uses TF-IDF features with Logistic Regression.

- Accuracy: 98.51%
- Precision: 98.58%
- Recall: 98.53%
- Macro-F1: 98.52%

### ONNX Sentence Embeddings

The project uses the `all-MiniLM-L6-v2` model for sentence embeddings and semantic search.

The model is exported to ONNX and runs using ONNX Runtime, allowing the embedding pipeline to run without the full PyTorch-based Sentence Transformers runtime.

The ONNX model produces 384-dimensional sentence embeddings.

The embedding-based classifier was evaluated on the held-out test set:

- Accuracy: 98.66%
- Precision: 98.71%
- Recall: 98.69%
- Macro-F1: 98.68%

## Semantic Search

The chatbot converts the user's query into a sentence embedding and compares it with embeddings from the training-data retrieval corpus using cosine similarity.

The most similar training example is selected and its corresponding support response is returned.

The retrieval corpus is built using training data only to avoid test-data leakage during evaluation.

### Retrieval Evaluation

Semantic retrieval was evaluated on the full held-out test set.

- Recall@1: 98.44%
- MRR: 99.73%

## Chatbot Features

- Intent classification
- Semantic search
- Cosine-similarity based retrieval
- Training-data-only retrieval corpus
- Similarity threshold for unknown queries
- Fallback response for low-confidence queries
- Interactive command-line chatbot

## Project Files

- `ecommerce_support_chatbot_ipynb.ipynb` — Complete project notebook
- `app.py` — FastAPI chatbot application
- `model.onnx` — ONNX version of the `all-MiniLM-L6-v2` embedding model
- `tokenizer.json` — Tokenizer used with the ONNX model
- `ecommerce_tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `ecommerce_intent_classifier.pkl` — Trained intent classifier
- `ecommerce_retrieval_embeddings.npz` — Retrieval embeddings
- `ecommerce_retrieval_data.csv.gz` — Retrieval data
- `requirements.txt` — Required packages

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- ONNX Runtime
- Hugging Face Tokenizers
- FastAPI

## Deployment

The project is maintained on GitHub and designed for deployment using FastAPI and Vercel.

The ONNX model used for semantic search is approximately 86 MB.

## Authors

Ansari Iqra
