# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:30:07 2025

@author: H0sseini
"""
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.insert(1, '../Backend/')
from main import SummarizationTool, download_bart_large_cnn
import os

app = FastAPI()
if not download_bart_large_cnn(local_dir='../Backend/models/bart-large-cnn'):
    raise HTTPException(status_code=500, 
                       detail="Model file missing and failed to download.")
else:
    tool = SummarizationTool()

templates = Jinja2Templates(directory="./templates")
app.mount("/static", StaticFiles(directory="./templates/static"), name="static")

# Enable CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "error": None})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(
    request: Request,
    mode: str = Form(...),
    summary_type: str = Form(...),
    text_input: str = Form(None),
    file: UploadFile = File(None)
):
    
    if not download_bart_large_cnn(local_dir='../Backend/models/bart-large-cnn'):
        raise HTTPException(status_code=500, 
                           detail="Model file missing and failed to download.")
        
    try:
        # Use text input if available
        if text_input and text_input.strip():
            extracted_text = text_input.strip()
        elif file and file.filename:
            content = await file.read()
            extracted_text = tool.extract_text_from_bytes(content, file.filename)
        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "result": None,
                "error": "Please provide either a file or paste some text."
            })

        result = tool.summarize(extracted_text, mode, summary_type)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,
            "error": None
        })

    except Exception as e:
        import traceback
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": None,
            "error": str(e)
        })