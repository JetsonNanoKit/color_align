#!/usr/bin/env python3
"""
Color Align & Evaluation Web Service.
FastAPI backend providing:
- ZIP upload and extraction
- High-performance perceptual color difference statistics
- Adaptive color alignment & harmonization (ADMH / Global CDF / Reinhard)
- Zipped aligned image export and download
- Interactive web visualizer hosting
"""

import os
import shutil
import uuid
import zipfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.pipeline import ColorPipeline, match_image_pairs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

os.makedirs(SESSIONS_DIR, exist_ok=True)

app = FastAPI(
    title="Color Align & Evaluation System",
    description="Professional color difference statistics and adaptive color space alignment",
    version="1.0.0",
)

# Enable CORS for cross-origin or local usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset folders
if os.path.exists(os.path.join(WEB_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")

app.mount("/sessions", StaticFiles(directory=SESSIONS_DIR), name="sessions")


@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def serve_index():
    """Serves the primary web user interface."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index HTML not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/default_data")
@app.head("/api/default_data")
async def get_default_data():
    """Returns pre-evaluated demonstration dataset ('八字刘海' 70 pairs)."""
    data_path = os.path.join(WEB_DIR, "data.json")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Default demo data not found")
    with open(data_path, "r", encoding="utf-8") as f:
        import json
        return JSONResponse(content=json.load(f))


@app.post("/api/upload_and_align")
async def upload_and_align(
    src_file: UploadFile = File(..., description="Source image archive (src.zip)"),
    tag_file: UploadFile = File(..., description="Target image archive (tag.zip)"),
    mode: str = Form("admh", description="Alignment mode: 'admh', 'global_cdf', or 'reinhard'"),
):
    """
    Accepts src.zip and tag.zip, runs color discrepancy evaluation,
    performs color space alignment, and generates output archive.
    """
    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    upload_src_zip = os.path.join(session_dir, "src.zip")
    upload_tag_zip = os.path.join(session_dir, "tag.zip")
    raw_src_dir = os.path.join(session_dir, "raw_src")
    raw_tag_dir = os.path.join(session_dir, "raw_tag")

    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(raw_src_dir, exist_ok=True)
    os.makedirs(raw_tag_dir, exist_ok=True)

    try:
        # 1. Save uploaded zip files
        with open(upload_src_zip, "wb") as f:
            shutil.copyfileobj(src_file.file, f)

        with open(upload_tag_zip, "wb") as f:
            shutil.copyfileobj(tag_file.file, f)

        # 2. Extract ZIP files
        try:
            with zipfile.ZipFile(upload_src_zip, "r") as zf:
                zf.extractall(raw_src_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to unzip src.zip: {str(e)}")

        try:
            with zipfile.ZipFile(upload_tag_zip, "r") as zf:
                zf.extractall(raw_tag_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to unzip tag.zip: {str(e)}")

        # 3. Check for matching pairs
        pairs = match_image_pairs(raw_src_dir, raw_tag_dir)
        if not pairs:
            raise HTTPException(
                status_code=400,
                detail="No matching image filenames found between src.zip and tag.zip. Please ensure corresponding files share identical base names (e.g. 00001.png in both).",
            )

        # 4. Run Color Pipeline
        pipeline = ColorPipeline(mode=mode)
        result = pipeline.process(
            src_dir=raw_src_dir,
            tag_dir=raw_tag_dir,
            output_dir=session_dir,
            session_id=session_id,
            save_aligned=True,
            create_zip=True,
        )

        return {
            "status": "success",
            "session_id": session_id,
            "mode": mode,
            "summary": result["summary"],
            "items": result["items"],
            "download_url": f"/api/download_aligned/{session_id}",
            "report_url": f"/api/download_report/{session_id}",
            "duration": result["duration"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")


@app.get("/api/download_aligned/{session_id}")
@app.head("/api/download_aligned/{session_id}")
async def download_aligned(session_id: str):
    """Downloads the aligned image package (aligned_src.zip)."""
    zip_path = os.path.join(SESSIONS_DIR, session_id, "aligned_src.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Aligned package not found for this session.")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"aligned_src_{session_id}.zip",
    )


@app.get("/api/download_report/{session_id}")
@app.head("/api/download_report/{session_id}")
async def download_report(session_id: str):
    """Downloads the comprehensive evaluation report (data.json)."""
    json_path = os.path.join(SESSIONS_DIR, session_id, "data.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Report not found for this session.")
    return FileResponse(
        json_path,
        media_type="application/json",
        filename=f"color_report_{session_id}.json",
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting Color Alignment Web Application on http://localhost:8088 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8088, reload=False)
