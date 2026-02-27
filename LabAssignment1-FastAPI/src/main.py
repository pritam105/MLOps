from fastapi import FastAPI, status
from pydantic import BaseModel

from .predict import predict_text

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(req: TextRequest):
    return predict_text(req.text)