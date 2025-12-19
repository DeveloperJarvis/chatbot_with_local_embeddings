# 📘 Low-Level Design (LLD): Local-Embedding Chatbot (Non-ML)

## 1. System Overview

The **Local Embedding Chatbot** is a lightweight, offline chatbot that answers user queries by comparing them with a predefined knowledge base using **vector similarity** (cosine similarity).
No external machine learning frameworks or cloud APIs are used.

The chatbot operates in two modes:

- **Rule-based responses** (exact or pattern matches)
- **Embedding-based responses** (semantic similarity via vector math)

---

## 2. Key Objectives

- Fully offline operation
- No ML frameworks (no TensorFlow, PyTorch, etc.)
- Deterministic, explainable behavior
- Efficient similarity search using cosine similarity
- Simple NLP preprocessing

---

## 3. High-Level Architecture

```
User Input
   ↓
Text Preprocessor
   ↓
Rule Engine ──► (Exact Match Response)
   ↓ (fallback)
Embedding Engine
   ↓
Similarity Engine (Cosine Similarity)
   ↓
Response Selector
   ↓
Chatbot Output
```

---

## 4. Core Components

---

### 4.1 TextPreprocessor

**Responsibility:**
Normalize and tokenize user input and knowledge base text.

**Attributes:**

- stop_words
- punctuation_set

**Methods:**

- `normalize(text)`
- `tokenize(text)`
- `remove_stopwords(tokens)`
- `preprocess(text)`

**Techniques Used:**

- Lowercasing
- Tokenization (whitespace-based)
- Stopword removal
- Optional stemming (rule-based)

---

### 4.2 RuleEngine

**Responsibility:**
Handle deterministic, high-priority responses.

**Attributes:**

- rules (dictionary or list of patterns)

**Methods:**

- `match(text)`
- `get_response(text)`

**Examples:**

- Greetings (“hi”, “hello”)
- FAQs (“what is your name?”)
- Exit commands (“quit”, “bye”)

Rule-based responses override embedding logic.

---

### 4.3 KnowledgeBase

**Responsibility:**
Stores questions and their corresponding responses.

**Attributes:**

- entries (list of Q-A pairs)
- processed_entries

**Methods:**

- `load_data(source)`
- `get_questions()`
- `get_responses()`

**Data Format Example:**

```
[
  { "question": "What is Python?", "answer": "Python is a programming language." }
]
```

---

### 4.4 EmbeddingEngine (Non-ML)

**Responsibility:**
Convert text into numeric vectors using deterministic techniques.

**Embedding Strategies (Non-ML):**

- Bag-of-Words (BoW)
- TF-IDF (manual implementation)
- Character n-grams
- Hash-based vectors

**Attributes:**

- vocabulary
- vector_size

**Methods:**

- `build_vocabulary(texts)`
- `vectorize(text)`
- `vectorize_corpus(texts)`

No training phase; embeddings are computed directly.

---

### 4.5 SimilarityEngine

**Responsibility:**
Compute similarity between vectors.

**Methods:**

- `cosine_similarity(vec1, vec2)`
- `find_best_match(query_vector, corpus_vectors)`

**Formula Used:**

```
cosine_similarity = (A · B) / (||A|| × ||B||)
```

---

### 4.6 ResponseSelector

**Responsibility:**
Decide the final chatbot response.

**Attributes:**

- similarity_threshold

**Methods:**

- `select(best_match, score)`
- `fallback_response()`

**Logic:**

- If similarity ≥ threshold → return matched answer
- Else → return fallback (“I don’t understand”)

---

### 4.7 ChatbotController

**Responsibility:**
Orchestrates all components.

**Attributes:**

- preprocessor
- rule_engine
- embedding_engine
- similarity_engine
- knowledge_base

**Methods:**

- `handle_input(user_text)`
- `run_chat_loop()`

---

## 5. Data Flow

```
User Input
   ↓
Preprocess Text
   ↓
RuleEngine (Exact Match?)
   ↓ No
Vectorize Input
   ↓
Cosine Similarity with Knowledge Base
   ↓
Best Match + Score
   ↓
Response Selection
   ↓
Output
```

---

## 6. Error Handling & Edge Cases

- Empty input → prompt user again
- No similarity above threshold → fallback message
- Unknown words → zero-vector handling
- Division by zero in cosine similarity avoided

---

## 7. Performance Considerations

- Precompute embeddings for knowledge base
- Cache vectors to avoid recomputation
- Use simple Python lists/dicts for small datasets
- Suitable for hundreds to thousands of Q-A pairs

---

## 8. Extensibility

Possible future enhancements:

- Synonym expansion (manual dictionary)
- Context tracking (last N messages)
- Weighted keywords
- Phrase-level matching
- Persistent storage (JSON)

---

## 9. Example Interaction

```
User: What is Python?
Bot: Python is a programming language.

User: Hi
Bot: Hello! How can I help you?

User: Explain Java
Bot: I’m not sure about that yet.
```

---

## 10. Skills Demonstrated

✔ NLP preprocessing
✔ Vector math
✔ Cosine similarity
✔ Rule-based systems
✔ Clean architecture
✔ Offline systems

---

## 11. Interview One-Line Summary

> A fully offline Python chatbot that uses rule-based logic and local text embeddings with cosine similarity to deliver explainable, deterministic responses.
