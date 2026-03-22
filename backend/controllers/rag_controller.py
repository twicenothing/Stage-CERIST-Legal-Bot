from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
# Adjust this import path if your setup file is named differently!
from services.rag_setup import get_legal_answer

# Create the router for this specific feature
router = APIRouter(
    prefix="/rag",
    tags=["Legal Assistant"]
)

# Define the exact JSON shape we expect from the frontend
class ChatRequest(BaseModel):
    query: str

# The FastAPI Way: The decorator sits right on top of the logic
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"\n[API] Nouvelle question reçue: {request.query}")
    try:
        # Pass the extracted string to your RAG service
        result = await get_legal_answer(request.query)
        return result
    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))