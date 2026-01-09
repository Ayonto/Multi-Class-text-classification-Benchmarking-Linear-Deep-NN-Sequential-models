# Multi-Class Text Classification: Benchmarking Linear, Deep NN & Sequential Models

This repository contains the code, data, and report for a comprehensive benchmarking study of nine distinct machine learning and deep learning models for multi-class text classification, implemented for CSE440 course, BRAC University . The project evaluates the performance of various architectures—from Logistic Regression to Bidirectional GRUs and LSTMs—using both sparse (TF‑IDF) and dense (Word2Vec) text representations.

## 📄 Project Report

The full project report is available as a PDF:  
[`Multi-Class text classification Benchmarking.pdf`](report/Multi-Class%20text%20classification%20Benchmarking.pdf)

## Authors

- **Md. Shakibul Islam** – md.shakibul.islam1@g.bracu.ac.bd  
- **Mehedi Hasan** – mehedi.hasan12@g.bracu.ac.bd  
- **Junaed Ahmed** – junaed.ahmed1@g.bracu.ac.bd  

*Department of Computer Science & Engineering, BRAC University, Dhaka, Bangladesh*

## Abstract

This report evaluates nine distinct architectures for multi-class text classification using a balanced dataset of 93,333 open-domain user queries. We compare the efficacy of sparse TF‑IDF features against dense Word2Vec (Skip‑gram) embeddings across models ranging from Logistic Regression to Bidirectional LSTMs and GRUs. Experimental results confirm the critical advantage of semantic representations, with Word2Vec-based models consistently outperforming frequency-based baselines. The Bidirectional GRU emerged as the optimal architecture, achieving the highest Test Accuracy (**69.39%**) and Macro F1‑score (**0.6879**). While bidirectional sequence modeling significantly improved separation for distinct topics, error analysis reveals that static embeddings remain limited in resolving ambiguities between semantically overlapping categories.

## Dataset

- **Size**: 93,333 samples
- **Classes**: 10 balanced topics (each ~9.8–10.1% of data)
- **Content**: Open-domain user queries (QA text)
- **Preprocessing**: HTML tag removal, tokenization, lowercasing, stopword removal, lemmatization

## Models Implemented

We implemented and compared the following models:

| Model | Representation |
|-------|----------------|
| Logistic Regression | TF‑IDF |
| Deep Neural Network (DNN) | TF‑IDF |
| Deep Neural Network (DNN) | Word2Vec (Skip‑gram) |
| Simple RNN | Word2Vec |
| Bidirectional RNN | Word2Vec |
| Simple GRU | Word2Vec |
| **Bidirectional GRU** | **Word2Vec** |
| Simple LSTM | Word2Vec |
| Bidirectional LSTM | Word2Vec |

## Results

The Bidirectional GRU with Word2Vec embeddings achieved the best performance:

| Model | Representation | Macro F1‑Score | Accuracy |
|-------|----------------|----------------|----------|
| Bidirectional GRU | Word2Vec | **0.6879** | **69.39%** |

Detailed results for all models are available in the report and notebook.

## Dependencies 
The project uses the folowing key libraries: 
* Python 3.8+
* NumPy
* Pandas
* Scikit‑learn
* TensorFlow / Keras
* Gensim
* Matplotlib
* Seaborn
* NLTK

## License
This project is for academic and research purposes. Please cite the authors if you use the code or findings.
