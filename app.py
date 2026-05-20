# ========== app.py — ВЕБ-СЕРВЕР И API-ИНТЕРФЕЙС ==========
# Использует FastAPI для создания веб-приложения. Предоставляет эндпоинт /api/chat 
# для обработки запросов от пользователя и отдает HTML-интерфейс для работы с помощником.
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_simple import RobustRAG
import uvicorn

app = FastAPI(title="Помощник службы авиационной безопасности")

# Инициализируем RAG при старте сервера
rag = RobustRAG()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    # Получаем ответ от нашей RAG системы
    answer = rag.ask(request.question)
    return ChatResponse(answer=answer)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    # Отдаем фронтенд интерфейс
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Файл index.html не найден")
    
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print("Запуск веб-сервера RAG-помощника...")
    uvicorn.run(app, host="127.0.0.1", port=8000)