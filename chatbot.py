# --------------------------------------------------
# -*- Python -*- Compatibility Header
#
# Copyright (C) 2023 Developer Jarvis (Pen Name)
#
# This file is part of the chatbot_with_local_embeddings Library. This library is free
# software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# chatbot_with_local_embeddings - Rule-based or embedding-based chatbot using cosine similarity
#                               No external ML framework required
#                               Skills: NLP basics, vector math
#
# Author: Developer Jarvis (Pen Name)
# Contact: https://github.com/DeveloperJarvis
#
# --------------------------------------------------

# --------------------------------------------------
# chatbot_with_local_embeddings MODULE
# --------------------------------------------------

# --------------------------------------------------
# imports
# --------------------------------------------------

import json
import math
import string

# --------------------------------------------------
# 1. text preprocessor
# --------------------------------------------------
"""
Reponsibility:
Normalize and tokenize user input and knowledge base text.
Techniques Used:
- Lowercasing
- Tokenization (whitespace-based)
- Stopword removal
- Optional stemming (rule-based)
"""
class TextPreprocessor:
    def __init__(self):
        self.stop_words = {
            "I", "a", "an", "is", "in", "of", "to", "or", "on",
            "the", "and", "for", "how", "what", "with"
        }
        self.punctuation_set = set(string.punctuation)
    
    def normalize(self, text):
        return text.lower()

    def tokenize(self, text):
        return text.split()

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def remove_puncuations(self, tokens):
        return [
            "".join(c for c in token if c not in self.punctuation_set)
            for token in tokens
        ]

    def preprocess(self, text):
        text = self.normalize(text)
        tokens = self.tokenize(text)
        tokens = self.remove_puncuations(tokens)
        tokens = self.remove_stopwords(tokens)
        return [t for t in tokens if t]


# --------------------------------------------------
# 2. rule engine
# --------------------------------------------------
"""
Responsibility:
Handle deterministic, high-priority responses.
Examples:
- Greetings ("hi", "hello")
- FAQs ("what is your name?")
- Exit commands ("bye", "quit")
Rule-based responses override embedding logic.
"""
class RuleEngine:
    def __init__(self):
        self.rules = {
            "hi": "Hello! How can I help you?",
            "hello": "Hi there! What can I do for you?",
            "bye": "Goodbye! Have a nice day",
            "exit": "Goodbye! Have a nice day",
            "quit": "Goodbye! Have a nice day"
        }
    
    def match(self, text):
        return text.lower() in self.rules

    def get_response(self, text):
        return self.rules.get(text.lower())

# --------------------------------------------------
# 3. knowledge base
# --------------------------------------------------
"""
Responsibility:
Stores questions and their corresponding responses.
Data format example:
[
    {
        "question": "What is python?",
        "answer": "Python is a programming language"
    }
]
"""
class KnowledgeBase:
    def __init__(self, file_path):
        self.entries = []
        self.load_data(file_path)
    
    def load_data(self, source):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.entries = data.get("knowledge_base", [])

    def get_questions(self):
        return [e["question"] for e in self.entries]

    def get_answers(self):
        return [e["answer"] for e in self.entries]

# --------------------------------------------------
# 4. embedding engine (Non-ML)
# --------------------------------------------------
"""
Responsibility:
Convert text into numeric vectors using deterministic
techiniques.
Embedding Stategies (Non-ML):
- Bag_of_Words (BoW)
- TF-IDF (manual implementation)
- Character n-grams
- Hash-based vectors
No training phase; embeddings are computed directly
"""
class EmbeddingEngine:
    def __init__(self):
        self.vocabulary = {}

    def build_vocabulary(self, texts):
        index = 0
        for text in texts:
            for token in text:
                if token not in self.vocabulary:
                    self.vocabulary[token] = index
                    index += 1

    def vectorize(self, tokens):
        vector = [0] * len(self.vocabulary)
        for token in tokens:
            if token in self.vocabulary:
                vector[self.vocabulary[token]] += 1
        return vector

    def vectorize_corpus(self, corpus):
        return [self.vectorize(text) for text in corpus]

# --------------------------------------------------
# 5. similarity engine
# --------------------------------------------------
"""
Responsibility:
Compute similiarity between vectors
Formula Used:
cosine_similarity = (A - B) / (||A|| x ||B||)
"""
class SimilarityEngine:
    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def find_best_match(self, query_vector, corpus_vectors):
        best_score = 0.0
        best_index = -1

        for i, vec in enumerate(corpus_vectors):
            score = self.cosine_similarity(query_vector, vec)
            if score > best_score:
                best_score = score
                best_index = i
        
        return best_index, best_score


# --------------------------------------------------
# 6. response sector
# --------------------------------------------------
"""
Responsibility:
Decide the final chatbot response
Logic:
- If similarity >= threshold -> return matched answer
- Else -> return fallback ("I don't understand")
"""
class RespponseSelector:
    def __init__(self, threshold=0.25):
        self.similarity_threshold = threshold

    def select(self, index, score, answers):
        if index == -1 or score < self.similarity_threshold:
            return self.fallback_response()
        return answers[index]

    def fallback_response(self):
        return "I'm not sure about that yet."


# --------------------------------------------------
# 7. chatbot controller
# --------------------------------------------------
"""
Responsibility:
Orchestrates all components
"""
class ChatbotController:
    def __init__(self, kb_file):
        self.preprocessor = TextPreprocessor()
        self.rule_engine = RuleEngine()
        self.knowledge_base = KnowledgeBase(kb_file)
        self.embedding_engine = EmbeddingEngine()
        self.similarity_engine = SimilarityEngine()
        self.response_selector = RespponseSelector()

        self._prepare_embeddings()

    def _prepare_embeddings(self):
        processed_questions = [
            self.preprocessor.preprocess(q)
            for q in self.knowledge_base.get_questions()
        ]

        self.embedding_engine.build_vocabulary(processed_questions)
        self.corpus_vectors = self.embedding_engine.vectorize_corpus(
            processed_questions
        )

    def handle_input(self, user_text):
        if not user_text.strip():
            return "Please enter something"
        
        # Rule-based priority
        if self.rule_engine.match(user_text):
            return self.rule_engine.get_response(user_text)
        
        # Embedding-based
        tokens = self.preprocessor.preprocess(user_text)
        query_vector = self.embedding_engine.vectorize(tokens)

        index, score = self.similarity_engine.find_best_match(
            query_vector, self.corpus_vectors
        )

        return self.response_selector.select(
            index, score, self.knowledge_base.get_answers()
        )

    def run_chat_loop(self):
        print("Chatbot started. Type 'bye' to exit\n")

        while True:
            user_input = input("You: ")
            response = self.handle_input(user_input)
            print("Bot:", response)

            if user_input.lower() in {"bye", "exit", "quit"}:
                break


# --------------------------------------------------
# main module
# --------------------------------------------------
def main():
    chatbot = ChatbotController("knowledge_base.json")
    chatbot.run_chat_loop()

if __name__ == "__main__":
    main()
