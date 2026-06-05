import pickle
import re
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self):
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.index: BM25Okapi | None = None

    def build(self, ids: List[str], texts: List[str]) -> None:
        self.ids = ids
        self.texts = texts
        self.index = BM25Okapi([self._tokenize(t) for t in texts])

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        scores = self.index.get_scores(self._tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "texts": self.texts}, f)
        with open(path.with_suffix(".bm25.pkl"), "wb") as f:
            pickle.dump(self.index, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.ids = data["ids"]
        store.texts = data["texts"]
        with open(path.with_suffix(".bm25.pkl"), "rb") as f:
            store.index = pickle.load(f)
        return store

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
