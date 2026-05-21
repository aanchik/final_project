# ========== rag_simple.py — ЯДРО RAG-СИСТЕМЫ И ГЕНЕРАЦИЯ ОТВЕТОВ ==========
# Реализует поиск похожих фрагментов в базе и отправку контекста в LLM (Ollama)
# для формирования точных экспертных ответов.

# ========== ИМПОРТ БИБЛИОТЕК ========== 
# Подключение ChromaDB, HTTP-запросов и модели эмбеддингов.
import chromadb
import requests
from sentence_transformers import SentenceTransformer
from config import DB_DIR, EMBEDDING_MODEL_NAME, TOP_K_RETRIEVAL

# ========== КЛАСС RAG-СИСТЕМЫ ========== 
# Основное ядро системы для поиска и генерации ответов через LLM.
class RobustRAG:
    def __init__(self):
        # Подключение к ChromaDB, загрузка эмбеддингов и настройка Ollama.
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_collection(name="sab_docs")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # ========== НАСТРОЙКА LLM ========== 
        # Указание адреса Ollama и модели для генерации ответов.
        self.ollama_url = "http://localhost:11434"
        self.model = "qwen2.5:3b-instruct-q4_K_M"
        
        # Шаблон запроса к LLM с инструкциями, примерами и правилами ответа.
        self.prompt_template = """Ты — старший эксперт Службы авиационной безопасности (САБ) в аэропорту. 
Ответь на внутренний рабочий вопрос сотрудника КПП/досмотра, строго сопоставляя факты из предоставленных документов.

Инструкция по логике ответов:
1. Сравнивай числа: если в документе лимит "до 5 кг", а в вопросе "7 кг" — отвечай, что это запрещено, так как превышает лимит.
2. Понимай синонимы: "в одном кейсе / вместе" означает, что предметы лежат НЕ отдельно; "пистолет / ружье" — это оружие.
3. Обязательно пиши название файла-источника в конце ответа в формате [название_файла.txt].

ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ ДЛЯ ТЕБЯ:
Контекст: [оружие.txt] Патроны весом более 5 кг к перевозке не допускаются. Патроны к оружию размещаются отдельно от оружия.
Вопрос: Пассажир заявляет, что у него 6 кг патронов. Можно пропустить?
Ответ: Нет, нельзя. Согласно документу [оружие.txt], максимальный разрешенный вес патронов составляет 5 кг. Вес в 6 кг превышает установленную норму.

Контекст: [досмотр.txt] Лица, обладающие дипломатическим иммунитетом, досмотру не подлежат.
Вопрос: Обязан ли дипломат проходить проверку на рамке на общих основаниях?
Ответ: Нет, не обязан. Согласно [досмотр.txt], лица с дипломатическим статусом и иммунитетом освобождаются от досмотра.

РАБОЧЕЕ ЗАДАНИЕ:
Контекст:
{context}

Вопрос: {question}
Ответ:"""
    # ========== МЕТОД RETRIEVE ==========
    # Поиск релевантных фрагментов документов из векторной базы.
    def retrieve(self, query, k=TOP_K_RETRIEVAL):
        query_text = f"query: {query}"
        # Преобразование запроса пользователя в вектор.
        query_emb = self.embed_model.encode(query_text).tolist()
        
        # Поиск наиболее похожих документов по эмбеддингу.
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k
        )
        
        chunks = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                chunks.append((doc, meta))
        return chunks
    
    # ========== ОСНОВНОЙ МЕТОД ОТВЕТА (ASK) ==========
    # Главная функция RAG: получает вопрос, ищет контекст и возвращает ответ
    def ask(self, query):
        chunks = self.retrieve(query)
        if not chunks:
            return "Информация в нормативной базе отсутствует."
        
        # Объединение найденных документов в единый текст для LLM.
        context = "\n\n".join(f"[{m['source']}] {c}" for c, m in chunks)
        prompt = self.prompt_template.format(context=context, question=query)
        
        try:
            # Отправка промпта в локальную LLM и получение ответа.
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model, 
                    "prompt": prompt, 
                    "stream": False, 
                    "options": {
                        "temperature": 0.0,  # Полное отсутствие отсебятины
                        "top_p": 0.1
                    }
                },
                timeout=60
            )
            # Возвращает финальный ответ пользователю или ошибку.
            return resp.json()["response"]
        except Exception as e:
            return f"Ошибка Ollama: {str(e)}"

# ========== CLI РЕЖИМ ТЕСТИРОВАНИЯ ==========
# Запуск системы в консольном режиме для ручной проверки.
if __name__ == "__main__":
    rag = RobustRAG()
    while True:
        q = input("\nВопрос САБ: ")
        if q.lower() == "exit": break
        print(rag.ask(q))