from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import shutil
import tempfile
import os

from vllm import LLM
from ocrflux.inference import parse

MODEL_PATH = "/model_dir/OCRFlux-3B"  # Change to your actual model path

app = FastAPI()
llm = None

@app.on_event("startup")
def load_model():
    global llm
    if llm is None:
        llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.8, max_model_len=8192)

@app.post("/ocr/", response_class=PlainTextResponse)
async def ocr_file(file: UploadFile = File(...)):
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = parse(llm, tmp_path)
        if result is None:
            raise HTTPException(status_code=500, detail="OCR parsing failed.")
        return result["document_text"]
    finally:
        os.remove(tmp_path)
