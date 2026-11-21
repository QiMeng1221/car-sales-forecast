
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from xgboost import XGBRegressor

# Load the pre-trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "model.json")

model = XGBRegressor()
model.load_model(model_path)

# Initialize FastAPI app
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class SalesInput(BaseModel):
    sales: list[float] = Field(min_length=6, max_length=6)

# Mount frontend directory
frontend_dir = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Serve HTML page at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# Prediction endpoint
@app.post("/predict")
async def predict(input: SalesInput):

    # Validate input values
    if not all(np.isfinite(v) and v >= 0 for v in input.sales):
        raise HTTPException(
            status_code=400,
            detail="Sales values must be finite and non-negative."
        )

    # Input order: lag_1, lag_2, ..., lag_6
    data = np.array(input.sales).reshape(1, -1)

    # Generate next-month forecast
    prediction = model.predict(data)[0]

    return {"prediction": round(float(prediction), 2)}
