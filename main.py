from fastapi import FastAPI
from chat_router import router as chat_router

app = FastAPI(title="AI E-commerce Chatbot")

app.include_router(chat_router)

@app.get("/")
async def health():
    return {"msg": "Chatbot backend is running!"}