from fastapi import FastAPI 
from contextlib import asynccontextmanager
from services.rag_setup import init_rag
from api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_rag()
    yield
    print("shutting down..")


app = FastAPI(lifespan = lifespan)
app.include_router(api_router,prefix="/api")


@app.get("/")
def hello_world():
    return{"hello":"world"}