from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import json
from text_extraction import extract_text_from_pdf
from pdf_llm_agent_pipeline import run_agent_on_text

app = FastAPI()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), ground_truth: Optional[str] = Form(None)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        text = extract_text_from_pdf(tmp_path)
        gt_dict = json.loads(ground_truth) if ground_truth else None
        prediction = run_agent_on_text(text, gt_dict)
        return JSONResponse(content={"result": prediction})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload-text")
async def upload_text(text: str = Form(...), ground_truth: Optional[str] = Form(None)):
    try:
        gt_dict = json.loads(ground_truth) if ground_truth else None
        prediction = run_agent_on_text(text, gt_dict)
        return JSONResponse(content={"result": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})