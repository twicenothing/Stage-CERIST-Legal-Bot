from fastapi import APIRouter

# Import your new RAG controller
from controllers.rag_controller import router as rag_router

# Create the master API router
router = APIRouter()

# Plug the RAG controller into the master router
router.include_router(rag_router)

# As your app grows, you just add more lines here:
# router.include_router(auth_controller.router)
# router.include_router(user_controller.router)