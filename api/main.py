from fastapi import FastAPI
from .inference import _router as inference_router
from .drift import router as drift_router

app = FastAPI(
    title="Crypto ML API",
    version="1.0",
    description="API de prédiction crypto avec modèles ML"
)

# Routers
app.include_router(inference_router, tags=["inference"])
app.include_router(drift_router, tags=["drift"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
def root():
    return {"message": "Crypto ML API - use /docs for documentation"}
