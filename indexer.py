# ========== indexer.py — ИНДЕКСАЦИЯ ДОКУМЕНТОВ В CHROMADB ==========
# Читает сырые текстовые файлы, разбивает их на смысловые фрагменты (chunks)
# с помощью RecursiveCharacterTextSplitter и сохраняет в векторную базу данных.

# ========== ИМПОРТ БИБЛИОТЕК ==========
# Работа с файлами, ChromaDB и эмбеддингами для индексации документов.
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCS_DIR, DB_DIR, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

# ========== ОСНОВНАЯ ФУНКЦИЯ ИНДЕКСАЦИИ ==========
# Запуск процесса обработки документов и создания векторной базы.
def main():
    # Проверяем наличие директории с нормативными файлами.
    if not os.path.exists(DOCS_DIR):
        print(f"Ошибка: Папка {DOCS_DIR} не найдена!")
        return

    print("Загрузка модели эмбеддингов E5...")
    # Подготовка модели для преобразования текста в векторы.
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # ========== ИНИЦИАЛИЗАЦИЯ CHROMADB ========== 
    # Создание или пересоздание векторной базы данных.
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(name="sab_docs")
    except Exception:
        pass
    collection = client.create_collection(name="sab_docs")

    # Умный сплиттер под новые увеличенные размеры
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    docs, metadatas, ids, embeddings = [], [], [], []
    chunk_id_counter = 0

    # Обработка файлов. Чтение всех .txt файлов и извлечение текстовых фрагментов.
    print("Сканирование файлов и нарезка...")
    for fname in os.listdir(DOCS_DIR):
        if not fname.endswith(".txt"): continue
        
        with open(os.path.join(DOCS_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()

        # Разбиение текста на смысловые фрагменты.
        chunks = splitter.split_text(text)
        for chunk in chunks:
            passage_text = f"passage: {chunk}"
            docs.append(chunk) 
            metadatas.append({"source": fname})
            ids.append(f"id_{fname}_{chunk_id_counter}")
            # Преобразование текстовых фрагментов в векторы.
            embeddings.append(embed_model.encode(passage_text).tolist())
            chunk_id_counter += 1

    # ========== СОХРАНЕНИЕ В CHROMADB ========== 
    # Добавление документов, эмбеддингов и метаданных в базу.
    collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"[УСПЕХ] База ChromaDB пересоздана. Нарезано фрагментов: {len(docs)}")

# ========== ЗАПУСК СКРИПТА ========== 
# Точка входа для выполнения индексации.
if __name__ == "__main__":
    main()