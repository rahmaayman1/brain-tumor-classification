# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from src.inference import predict

app = FastAPI(
    title="Brain Tumor Classification API",
    description="Classifies brain MRI scans into 4 categories using EfficientNetB0",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Brain Tumor Classification API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG are supported."
        )

    # Read and process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Run inference
    result = predict(image)

    return {
        "filename": file.filename,
        "prediction": result
    }