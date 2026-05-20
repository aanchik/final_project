# ========== indexer.py — ИНДЕКСАЦИЯ ДОКУМЕНТОВ В CHROMADB ==========
# Читает сырые текстовые файлы, разбивает их на смысловые фрагменты (chunks)
# с помощью RecursiveCharacterTextSplitter и сохраняет в векторную базу данных.
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCS_DIR, DB_DIR, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def main():
    if not os.path.exists(DOCS_DIR):
        print(f"Ошибка: Папка {DOCS_DIR} не найдена!")
        return

    print("Загрузка модели эмбеддингов E5...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
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

    print("Сканирование файлов и нарезка...")
    for fname in os.listdir(DOCS_DIR):
        if not fname.endswith(".txt"): continue
        
        with open(os.path.join(DOCS_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        for chunk in chunks:
            passage_text = f"passage: {chunk}"
            docs.append(chunk) 
            metadatas.append({"source": fname})
            ids.append(f"id_{fname}_{chunk_id_counter}")
            embeddings.append(embed_model.encode(passage_text).tolist())
            chunk_id_counter += 1

    collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"[УСПЕХ] База ChromaDB пересоздана. Нарезано фрагментов: {len(docs)}")

if __name__ == "__main__":
    main()