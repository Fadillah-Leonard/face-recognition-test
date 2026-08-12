from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import numpy as np
import cv2

from database import init_db, insert_embedding, find_best_match, count_records
from face_model import FaceModel

model = FaceModel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    model.load()
    yield

app = FastAPI(
    title="Face Recognition Server",
    version="1.0.0",
    lifespan=lifespan
)

async def read_image(upload: UploadFile) -> np.ndarray:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="File kosong.")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Gambar tidak bisa dibaca.")

    return image


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "face-recognition-server",
        "records": count_records()
    }


@app.post("/register")
async def register(file: UploadFile = File(...)):
    image = await read_image(file)
    embedding = model.get_embedding(image)

    if embedding is None:
        raise HTTPException(status_code=400, detail="Tidak ditemukan tepat satu wajah.")

    new_id = insert_embedding(embedding)

    return JSONResponse({
        "success": True,
        "id": new_id
    })


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await read_image(file)
    embedding = model.get_embedding(image)

    if embedding is None:
        raise HTTPException(status_code=400, detail="Tidak ditemukan tepat satu wajah.")

    result = find_best_match(embedding)

    return JSONResponse({
        "success": True,
        **result
    })


@app.get("/health")
def health():
    return {"status": "healthy"}
