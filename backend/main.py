from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from utils.logger import setup_logger
import uvicorn
import os

logger = setup_logger("FastAPI_Backend")

app = FastAPI(title="Fraud Investigation API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the API and initializing database...")
    init_db()
    # Initialize ML predictor globally so it loads models on startup
    logger.info("Loading Machine Learning models into memory...")
    app.state.predictor = FraudPredictor()
    logger.info("FraudPredictor successfully loaded and ready.")

app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
