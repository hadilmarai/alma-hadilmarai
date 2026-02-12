import os
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class DocChunk:
    doc_id: str
    text: str

class MiniRAG:
    def __init__(self, docs: List[DocChunk]):
        if not docs:
            raise ValueError("No documents found for RAG.")

        texts = [d.text.strip() for d in docs if d.text.strip()]
        if not texts:
            raise ValueError("All RAG documents are empty.")

        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1
        )

        try:
            self.matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # Fallback: disable stopwords
            self.vectorizer = TfidfVectorizer(min_df=1)
            self.matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def from_folder(folder: str) -> "MiniRAG":
        docs = []

        if not os.path.exists(folder):
            raise ValueError(f"RAG folder '{folder}' does not exist.")

        for fname in os.listdir(folder):
            if fname.endswith(".md") or fname.endswith(".txt"):
                path = os.path.join(folder, fname)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    docs.append(DocChunk(doc_id=fname, text=text))

        return MiniRAG(docs)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[DocChunk, float]]:
        if not query.strip():
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()

        top_idx = sims.argsort()[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in top_idx]
