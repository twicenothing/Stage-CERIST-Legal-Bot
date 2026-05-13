from fastapi import APIRouter

# Import your new RAG controller
from controllers.rag_controller import router as rag_router
from controllers.user_controller import router as user_router
# Create the master API router
router = APIRouter()


router.include_router(rag_router)
router.include_router(user_router)

