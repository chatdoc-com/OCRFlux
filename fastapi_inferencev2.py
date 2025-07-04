from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
import shutil
import tempfile
import os
import logging
from typing import Optional
from vllm import LLM, SamplingParams
from ocrflux.inference import parse

MODEL_PATH = "/model_dir/OCRFlux-3B"  # Change to your actual model path
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="OCRFlux API",
    description="API for OCRFlux PDF-to-Markdown conversion with advanced layout handling",
    version="1.0.0"
)

llm = None

@app.on_event("startup")
def load_model():
    global llm
    logger.info(f"Loading model from {MODEL_PATH}")
    try:
        llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.8, max_model_len=8192)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
        
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
@app.post("/ocr/", response_class=PlainTextResponse)
async def ocr_file(
    file: UploadFile = File(...),
    skip_cross_page_merge: bool = Query(False, description="Skip cross-page content merging"),
    max_page_retries: int = Query(2, description="Maximum retries for page processing")
):
    """
    Process a PDF file and convert it to markdown text.
    - **file**: PDF file to process
    - **skip_cross_page_merge**: Whether to skip cross-page content merging
    - **max_page_retries**: Maximum number of retries for page processing
    """
    logger.info(f"Processing file: {file.filename}")
    # Validate file extension
    if not file.filename.lower().endswith(('.pdf')):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        # Call the OCRFlux parse function with parameters
        result = parse(
            llm,
            tmp_path,
            skip_cross_page_merge=skip_cross_page_merge,
            max_page_retries=max_page_retries
        )
        if result is None:
            raise HTTPException(status_code=500, detail="OCR parsing failed.")
        return result["document_text"]
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Removed temporary file: {tmp_path}")
@app.post("/ocr/full/", response_class=JSONResponse)
async def ocr_file_full(
    file: UploadFile = File(...),
    skip_cross_page_merge: bool = Query(False, description="Skip cross-page content merging"),
    max_page_retries: int = Query(2, description="Maximum retries for page processing")
):
    """
    Process a PDF file and return complete OCR results including:
    - document_text: The full markdown text
    - page_texts: Individual page texts
    - fallback_pages: Pages that couldn't be processed
    - num_pages: Total number of pages
    """
    logger.info(f"Processing file with full response: {file.filename}")
    # Validate file extension
    if not file.filename.lower().endswith(('.pdf')):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        # Call the OCRFlux parse function with parameters
        result = parse(
            llm,
            tmp_path,
            skip_cross_page_merge=skip_cross_page_merge,
            max_page_retries=max_page_retries
        )
        if result is None:
            raise HTTPException(status_code=500, detail="OCR parsing failed.")
        return result
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Removed temporary file: {tmp_path}")
@app.get("/health")
async def health_check():
    """Health check endpoint to verify if the service is running"""
    return {"status": "healthy", "model_loaded": llm is not None}




