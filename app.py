# ========== app.py — ВЕБ-СЕРВЕР И API-ИНТЕРФЕЙС ==========
# Использует FastAPI для создания веб-приложения. Предоставляет эндпоинт /api/chat 
# для обработки запросов от пользователя и отдает HTML-интерфейс для работы с помощником.

# ========== ИМПОРТ БИБЛИОТЕК ==========
# Подключение FastAPI, RAG-системы и инструментов запуска сервера.
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_simple import RobustRAG
import uvicorn

# ========== СОЗДАНИЕ FASTAPI ПРИЛОЖЕНИЯ ========== 
# Инициализация веб-приложения и настройка API.
app = FastAPI(title="Справочник Инспектора САБ")

# ========== ИНИЦИАЛИЗАЦИЯ RAG-СИСТЕМЫ ========== 
# Загрузка RAG-системы при запуске сервера.
rag = RobustRAG()

# ========== МОДЕЛЬ ЗАПРОСА ========== 
# Структура JSON-запроса от пользователя.
class ChatRequest(BaseModel):
    question: str

# ========== МОДЕЛЬ ОТВЕТА ========== 
# Структура JSON-ответа сервера.
class ChatResponse(BaseModel):
    answer: str

# ========== API ДЛЯ ОБРАБОТКИ ВОПРОСОВ ========== 
# Эндпоинт /api/chat принимает вопрос и возвращает ответ RAG-системы.
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    # Получаем ответ от нашей RAG системы
    answer = rag.ask(request.question)
    return ChatResponse(answer=answer)

# ========== ВЕБ-ИНТЕРФЕЙС ========== 
# Отправка HTML-страницы index.html пользователю.
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    # Отдаем фронтенд интерфейс
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Файл index.html не найден")
    
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# ========== ЗАПУСК СЕРВЕРА ========== 
# Запуск FastAPI сервера через Uvicorn на порту 8000.
if __name__ == "__main__":
    print("Запуск веб-сервера RAG-помощника...")
    uvicorn.run(app, host="0.0.0.0", port=8000)