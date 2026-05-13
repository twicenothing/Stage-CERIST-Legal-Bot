from fastapi import FastAPI 
from contextlib import asynccontextmanager
from services.rag_setup import init_rag
from api.routes import router as api_router
from sqlalchemy.sql import text
from core.database import engine, Base
from fastapi.middleware.cors import CORSMiddleware
@asynccontextmanager
async def lifespan(app: FastAPI):

    try :
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print(" SUCCESS: Connected to SQLite database!")
        
        Base.metadata.create_all(bind=engine)
        print("Database tables verified/created!")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to database. Details: {e}")
        raise e 
    


    await init_rag()
    yield
    print("shutting down..")


app = FastAPI(lifespan = lifespan)
app.include_router(api_router,prefix="/api")
# 1. Define the frontend URLs that are allowed to talk to your API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173", # Good practice to include both
    # "https://your-production-frontend.com" # Add your live URL here later
]

# 2. Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # MUST be True because your frontend uses credentials: "include"
    allow_methods=["*"],    # Allows all methods (GET, POST, PUT, DELETE, OPTIONS)
    allow_headers=["*"],    # Allows all headers (Crucial for your 'Authorization' zamili_token)
)


