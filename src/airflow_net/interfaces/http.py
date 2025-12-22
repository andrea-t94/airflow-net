from fastapi import FastAPI
from ..engine import ModelEngine

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Airflow-Net HTTP Server Running"}
