from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routers import history
from services.db_service import connect_db, close_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(
    title="ECG History Service",
    description="Recupero e gestione delle predizioni ECG storiche",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(history.router, prefix="/history", tags=["history"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "history-service"}