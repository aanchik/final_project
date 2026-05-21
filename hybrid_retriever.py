# ========== hybrid_retriever.py — ГИБРИДНЫЙ ПОИСК (FAISS + BM25) ==========
# Реализует комбинированный алгоритм поиска: семантический (FAISS, на векторах) 
# и ключевой (BM25, по словам). Это позволяет находить ответы точнее, 
# объединяя смысл фразы и точное совпадение терминов.

# ========== ИМПОРТ БИБЛИОТЕК ==========
# Подключение FAISS, BM25, эмбеддингов и утилит для работы с данными.
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
from config import INDEX_DIR, EMBEDDING_MODEL_NAME, TOP_K_RETRIEVAL

# ========== КЛАСС ГИБРИДНОГО ПОИСКА ========== 
# Основной класс, объединяющий BM25 (ключевой поиск) и FAISS (семантический поиск).
class HybridRetriever:
    def __init__(self):
        # Загружаем FAISS-индекс и данные
        self.index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
        with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        # Модель эмбеддингов
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embed_model.max_seq_length = 512

        # Строим BM25 индекс (токенизация по словам)
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    # ========== МЕТОД ГИБРИДНОГО ПОИСКА ==========
    # Основная функция поиска: комбинирует BM25 и FAISS.
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL, alpha: float = 0.6):
        """
        Гибридный поиск: BM25 + FAISS.
        alpha – вес BM25 (1-alpha – вес семантики).
        """
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        # Нормализация BM25 (min-max)
        if np.max(bm25_scores) - np.min(bm25_scores) > 0:
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        # FAISS (семантический) scores
        query_emb = self.embed_model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(query_emb).astype(np.float32), len(self.chunks))
        # Преобразуем расстояния в сходство (обратное расстояние)
        faiss_scores = 1.0 / (1.0 + distances[0])
        if np.max(faiss_scores) - np.min(faiss_scores) > 0:
            faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores))
        else:
            faiss_scores = np.zeros_like(faiss_scores)

        # Комбинируем
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        results = [(self.chunks[i], self.metadata[i]) for i in top_indices]
        return results