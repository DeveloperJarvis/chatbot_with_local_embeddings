# Chatbot with Local Embeddings (Non-ML)

## Overview

**Chatbot with Local Embeddings** is a lightweight, fully offline chatbot implemented in Python.
It uses **rule-based logic** combined with **local text embeddings** and **cosine similarity** to answer user queries—without relying on any external machine learning frameworks or cloud services.

The project focuses on **explainable NLP techniques**, deterministic behavior, and fundamental vector mathematics.

---

## Features

- Fully offline chatbot
- No external ML frameworks (no TensorFlow, PyTorch, etc.)
- Rule-based responses for high-priority queries
- Embedding-based semantic matching using cosine similarity
- Simple NLP preprocessing (tokenization, normalization)
- Deterministic and explainable responses
- Easy to extend with new knowledge entries

---

## Architecture Overview

```
User Input
   ↓
Text Preprocessor
   ↓
Rule Engine ──► Exact Match Response
   ↓ (fallback)
Embedding Engine
   ↓
Cosine Similarity
   ↓
Best Match Selection
   ↓
Chatbot Response
```

---

## Core Components

### 1. Text Preprocessor

- Lowercasing
- Tokenization
- Stopword removal
- Optional stemming (rule-based)

### 2. Rule Engine

- Handles greetings and fixed responses
- Takes priority over embeddings
- Ensures deterministic behavior

### 3. Knowledge Base

- Stores predefined questions and answers
- Loaded from local data (e.g., JSON)

### 4. Embedding Engine (Non-ML)

- Converts text into numeric vectors
- Techniques:

  - Bag-of-Words (BoW)
  - Manual TF-IDF
  - Character n-grams

- No training phase required

### 5. Similarity Engine

- Computes cosine similarity between vectors
- Finds best matching question

### 6. Response Selector

- Uses similarity threshold to determine confidence
- Falls back gracefully when confidence is low

---

## Example Interaction

```
User: Hi
Bot: Hello! How can I help you?

User: What is Python?
Bot: Python is a programming language.

User: Explain quantum computing
Bot: Sorry, I don't have an answer for that yet.
```

---

## Project Structure

```
chatbot_with_local_embeddings/
│
├── chatbot.py
├── knowledge_base.json
├── README.md
└── tests/
    └── test_similarity.py
```

---

## Requirements

- Python 3.8 or higher
- Standard library only
- No internet connection required

---

## Design Goals

- **Simplicity** – easy to understand and modify
- **Explainability** – no black-box models
- **Portability** – runs anywhere Python runs
- **Performance** – suitable for small to medium knowledge bases

---

## Extensibility

Possible enhancements:

- Context tracking (multi-turn conversations)
- Synonym expansion using local dictionaries
- Weighted keywords
- JSON/CSV knowledge base import
- CLI or GUI interface

---

## Limitations

- No deep semantic understanding
- Performance degrades with very large datasets
- Vocabulary must be maintained manually

---

## License

This project is licensed under the **GNU General Public License v3.0 or later**.
See the `LICENSE` file for details.

---

## Author

**Developer Jarvis (Pen Name)**
GitHub: [https://github.com/DeveloperJarvis](https://github.com/DeveloperJarvis)

---

## Interview Summary (One Line)

> A Python-based offline chatbot using rule-based logic and local embeddings with cosine similarity, demonstrating core NLP and vector math concepts without external ML frameworks.

## 🏷️Creating tag

```bash
# 1. Check existing tags
git tag
# 2. Create a valid tag
git tag -a v1.0.0 -m "Release version 1.0.0"
# or lightweight tag
git tag v1.0.0
# push tag to remote
git push origin v1.0.0
```
