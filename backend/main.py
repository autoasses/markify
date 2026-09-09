from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import pymupdf
import pymupdf4llm
import torch

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from PIL import Image

from olmocr_grading.config import OCRConfig, load_config
from olmocr_grading.image_utils import (
    SUPPORTED_IMAGE_EXTS,
    preprocess_image,
    resize_to_longest_dim,
)
from olmocr_grading.ocr_engine import OlmOCREngine, PageResult
from olmocr_grading.pdf_utils import pdf_to_images

from segmentation import segment_document

import db


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("autoassess")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="AutoAssess PDF Processing API")


# ============================================================
# CORS
# ============================================================

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OCR CONFIGURATION
# ============================================================

def get_ocr_dtype() -> str:
    requested = os.getenv("OCR_DTYPE")

    if requested:
        requested = requested.lower()

        valid = {"float32", "float16", "bfloat16"}

        if requested not in valid:
            raise ValueError(
                f"Invalid OCR_DTYPE='{requested}'. "
                f"Expected one of {sorted(valid)}."
            )

        if requested == "bfloat16":
            if not torch.cuda.is_available():
                logger.warning(
                    "BF16 requested but CUDA is unavailable. "
                    "Using float32."
                )
                return "float32"

            major, _ = torch.cuda.get_device_capability()

            if major < 8:
                logger.warning(
                    "BF16 requested but GPU does not support "
                    "native BF16. Using float16."
                )
                return "float16"

        return requested

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()

        if major >= 8:
            logger.info(
                "GPU compute capability %d.%d supports BF16. "
                "Using bfloat16.",
                major,
                minor,
            )
            return "bfloat16"

        logger.info(
            "GPU compute capability %d.%d does not support "
            "native BF16. Using float16.",
            major,
            minor,
        )
        return "float16"

    return "float32"


def build_ocr_config() -> OCRConfig:
    cfg = load_config(None)

    requested_device = os.getenv("OCR_DEVICE", "auto").lower()

    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            f"Invalid OCR_DEVICE='{requested_device}'. "
            "Expected auto, cuda, or cpu."
        )

    cuda_available = torch.cuda.is_available()

    if requested_device == "cpu":
        cfg.device = "cpu"
        cfg.dtype = "float32"

    elif requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "OCR_DEVICE=cuda was requested, "
                "but CUDA is unavailable."
            )

        cfg.device = "cuda"
        cfg.dtype = get_ocr_dtype()

    else:
        if cuda_available:
            cfg.device = "cuda"
            cfg.dtype = get_ocr_dtype()
        else:
            logger.warning(
                "CUDA is unavailable. Falling back to CPU."
            )
            cfg.device = "cpu"
            cfg.dtype = "float32"

    if cfg.device == "cuda":
        device_index = torch.cuda.current_device()

        logger.info(
            "CUDA device=%d: %s",
            device_index,
            torch.cuda.get_device_name(device_index),
        )

        logger.info(
            "CUDA capability=%s",
            torch.cuda.get_device_capability(device_index),
        )

        logger.info(
            "CUDA memory: %.2f GB total",
            torch.cuda.get_device_properties(device_index).total_memory
            / (1024 ** 3),
        )

    return cfg


OCR_CFG: OCRConfig = build_ocr_config()
OCR_ENGINE: Optional[OlmOCREngine] = None


# ============================================================
# OCR MODEL LIFECYCLE
# ============================================================

@app.on_event("startup")
def load_ocr_model():
    global OCR_ENGINE

    logger.info(
        "Loading OCR model on device=%s dtype=%s",
        OCR_CFG.device,
        OCR_CFG.dtype,
    )

    OCR_ENGINE = OlmOCREngine(OCR_CFG)
    OCR_ENGINE.load()

    if OCR_CFG.device == "cuda":
        torch.cuda.synchronize()

    logger.info("OCR model loaded successfully.")


@app.on_event("shutdown")
def shutdown_ocr_model():
    global OCR_ENGINE

    OCR_ENGINE = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MONGODB LIFECYCLE
# ============================================================

@app.on_event("startup")
def connect_to_mongo():
    db.connect()


@app.on_event("shutdown")
def disconnect_from_mongo():
    db.close()


# ============================================================
# PDF → MARKDOWN
# ============================================================

def pdf_to_markdown(contents: bytes) -> str:
    """
    Convert PDF bytes directly to Markdown.

    No temporary file is used here.
    """

    doc = pymupdf.open(
        stream=contents,
        filetype="pdf",
    )

    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


# ============================================================
# MARKDOWN CLEANING
# ============================================================

def clean_markdown(markdown: str) -> str:
    """
    Remove common PDF extraction artifacts.
    """

    markdown = re.sub(
        r"\*{3}\s*\n\s*Page\s+\d+\s*\n\s*\*{3}",
        "",
        markdown,
        flags=re.IGNORECASE,
    )

    markdown = re.sub(
        r"\*+\s*-\s*o\s*O\s*o\s*-\s*\*+",
        "",
        markdown,
        flags=re.IGNORECASE,
    )

    markdown = re.sub(
        r"\n{3,}",
        "\n\n",
        markdown,
    )

    return markdown.strip()


# ============================================================
# OCR HELPERS
# ============================================================

def load_pages_from_bytes(
    data: bytes,
    suffix: str,
    cfg: OCRConfig,
) -> List[Image.Image]:

    suffix = suffix.lower()

    if suffix == ".pdf":
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            images = pdf_to_images(
                tmp_path,
                target_longest_dim=cfg.target_longest_dim,
                render_dpi=cfg.pdf_render_dpi,
            )

            return images

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if suffix in SUPPORTED_IMAGE_EXTS:
        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

        image = resize_to_longest_dim(
            image,
            cfg.target_longest_dim,
        )

        images = [image]

    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Expected PDF or {sorted(SUPPORTED_IMAGE_EXTS)}."
        )

    if cfg.preprocess:
        images = [
            preprocess_image(
                image,
                deskew=cfg.deskew,
                denoise=cfg.denoise,
                enhance_contrast=cfg.enhance_contrast,
            )
            for image in images
        ]

    return images


def run_ocr(
    images: List[Image.Image],
    engine: OlmOCREngine,
) -> List[PageResult]:

    results: List[PageResult] = []

    for page_number, image in enumerate(images, start=1):
        try:
            result = engine.transcribe(
                image,
                page_number=page_number,
            )

        except Exception:
            logger.exception(
                "OCR failed on page %d",
                page_number,
            )

            result = PageResult(
                page=page_number,
                text="[UNCLEAR]",
                confidence=None,
            )

        results.append(result)

    return results


def ocr_document(
    contents: bytes,
    filename: str,
    preprocess: bool = False,
):
    """
    Run the complete OCR pipeline and combine all pages
    into one document.
    """

    if OCR_ENGINE is None:
        raise RuntimeError("OCR model is not loaded.")

    suffix = Path(filename).suffix.lower()

    req_cfg = (
        OCR_CFG.copy()
        if hasattr(OCR_CFG, "copy")
        else OCR_CFG
    )

    req_cfg.preprocess = preprocess

    pages = load_pages_from_bytes(
        contents,
        suffix,
        req_cfg,
    )

    start = time.time()

    results = run_ocr(
        pages,
        OCR_ENGINE,
    )

    if OCR_CFG.device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    text = "\n\n".join(
        result.text
        for result in results
    )

    return {
        "text": text,
        "num_pages": len(pages),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_page": round(
            elapsed / max(len(pages), 1),
            2,
        ),
    }


# ============================================================
# DECIDE WHETHER OCR IS NECESSARY
# ============================================================

def should_use_ocr(markdown: str) -> bool:
    """
    Use OCR when PyMuPDF produced little meaningful text.
    """

    if not markdown:
        return True

    meaningful_text = re.sub(
        r"[\s#*_|`>\-]+",
        " ",
        markdown,
    ).strip()

    return len(meaningful_text) < 100


# ============================================================
# MODEL ANSWER DETECTION
# ============================================================

def is_model_answer_format(markdown: str) -> bool:
    text = markdown.upper()

    strong_indicators = [
        "MARKING SCHEME",
        "EXPECTED OUTCOMES",
        "VALUE POINTS",
    ]

    weak_indicators = [
        "MARKS",
        "Q.N",
        "QUESTION",
        "ANSWER",
    ]

    strong_matches = sum(
        indicator in text
        for indicator in strong_indicators
    )

    weak_matches = sum(
        indicator in text
        for indicator in weak_indicators
    )

    return (
        strong_matches >= 1
        and weak_matches >= 1
    )


# ============================================================
# PROCESS MARKING SCHEME
# ============================================================

def process_marking_scheme(markdown: str):
    lines = markdown.splitlines()

    questions = []
    current_question = None

    for line in lines:
        line = line.strip()

        if not line.startswith("|"):
            continue

        columns = [
            column.strip()
            for column in line.split("|")
        ]

        columns = [
            column
            for column in columns
            if column
        ]

        if len(columns) < 3:
            continue

        question_number = columns[0]
        answer_text = columns[1]
        marks_text = columns[2]

        if "Q.N" in question_number.upper():
            continue

        if "EXPECTED OUTCOMES" in answer_text.upper():
            continue

        if "MARKS" in marks_text.upper():
            continue

        if all(
            re.fullmatch(r"[-:]+", column)
            for column in columns[:3]
        ):
            continue

        match = re.search(
            r"\d+",
            question_number,
        )

        if match:
            number = int(match.group())

            current_question = {
                "number": number,
                "points": [],
            }

            questions.append(current_question)

        elif current_question is None:
            continue

        answer_text = answer_text.replace("<br>", " ")
        answer_text = answer_text.replace("<br/>", " ")
        answer_text = answer_text.replace("<br />", " ")

        answer_text = re.sub(
            r"\s+",
            " ",
            answer_text,
        ).strip()

        marks_match = re.search(
            r"(\d+(?:½|\.5)?)",
            marks_text,
        )

        marks = ""

        if marks_match:
            marks = marks_match.group(1)

        current_question["points"].append(
            {
                "text": answer_text,
                "marks": marks,
            }
        )

    return questions


# ============================================================
# GENERATE MODEL ANSWER MARKDOWN
# ============================================================

def generate_final_markdown(questions):
    output = "# Model Answer Paper\n\n"

    for question in questions:
        number = question["number"]
        total_marks = 0

        for point in question["points"]:
            marks = point["marks"]

            try:
                value = float(
                    marks.replace("½", ".5")
                )
                total_marks += value

            except (ValueError, AttributeError):
                pass

        if total_marks.is_integer():
            total_marks = int(total_marks)

        output += (
            f"## Q{number}. Model Answer "
            f"[{total_marks} Marks]\n\n"
        )

        for index, point in enumerate(
            question["points"],
            start=1,
        ):
            text = point["text"]
            marks = point["marks"]

            if marks:
                output += (
                    f"### {index}. "
                    f"[{marks} Marks]\n\n"
                )
            else:
                output += f"### {index}.\n\n"

            output += f"{text}\n\n"

        output += "---\n\n"

    return output.strip()


# ============================================================
# MAIN FILE ENDPOINT
# ============================================================

@app.post("/file")
async def accept_file(
    myFile: UploadFile = File(...),
    preprocess: bool = Form(False),
):
    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not myFile.filename:
        return {
            "success": False,
            "message": "No file selected.",
        }

    filename = myFile.filename
    suffix = Path(filename).suffix.lower()

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    supported_files = {
        ".pdf"
    } | set(SUPPORTED_IMAGE_EXTS)

    if suffix not in supported_files:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported types: "
                f"{sorted(supported_files)}"
            ),
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    contents = await myFile.read()

    logger.info(
        "Received file=%s type=%s size=%d bytes",
        filename,
        myFile.content_type,
        len(contents),
    )

    # ========================================================
    # STEP 1: Normal PDF → Markdown
    # ========================================================

    markdown = ""
    extraction_method = "pymupdf"
    used_ocr = False
    ocr_metadata = None

    if suffix == ".pdf":
        try:
            markdown = pdf_to_markdown(contents)
            markdown = clean_markdown(markdown)

        except Exception:
            logger.exception(
                "PyMuPDF extraction failed."
            )
            markdown = ""

    # ========================================================
    # STEP 2: OCR fallback
    # ========================================================

    if should_use_ocr(markdown):
        logger.info(
            "Insufficient PDF text detected. "
            "Falling back to olmOCR."
        )

        try:
            ocr_result = ocr_document(
                contents,
                filename,
                preprocess=preprocess,
            )

            markdown = clean_markdown(
                ocr_result["text"]
            )

            extraction_method = "olmocr"
            used_ocr = True

            ocr_metadata = {
                "device": OCR_CFG.device,
                "dtype": OCR_CFG.dtype,
                "num_pages": ocr_result["num_pages"],
                "elapsed_seconds": (
                    ocr_result["elapsed_seconds"]
                ),
                "seconds_per_page": (
                    ocr_result["seconds_per_page"]
                ),
            }

        except Exception as e:
            logger.exception("OCR failed.")

            raise HTTPException(
                status_code=500,
                detail=f"OCR failed: {str(e)}",
            )

    # ========================================================
    # STEP 3: Detect document type
    # ========================================================

    model_answer = is_model_answer_format(
        markdown
    )

    # ========================================================
    # STEP 4: Model Answer / Marking Scheme
    # ========================================================

    if model_answer:
        logger.info(
            "Detected: Model Answer / Marking Scheme"
        )

        questions = process_marking_scheme(
            markdown
        )

        final_markdown = generate_final_markdown(
            questions
        )

        response = {
            "success": True,
            "filename": filename,
            "type": "model_answer",
            "extraction_method": extraction_method,
            "used_ocr": used_ocr,
            "markdown": final_markdown,
            "questions": questions,
            "ocr": ocr_metadata,
        }

        document_id = db.save_document(
            response,
            db.SCANNED_DOCUMENTS_COLLECTION if used_ocr else db.MODEL_ANSWERS_COLLECTION,
        )
        response["document_id"] = document_id

        return response

    # ========================================================
    # STEP 5: Normal document → Question segmentation
    # ========================================================

    logger.info("Detected: Normal document")

    try:
        segments = segment_document(markdown)

    except Exception as e:
        logger.exception(
            "Segmentation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Segmentation failed: {str(e)}"
            ),
        )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    response = {
        "success": True,
        "filename": filename,
        "type": "raw",
        "extraction_method": extraction_method,
        "used_ocr": used_ocr,
        "markdown": markdown,
        "segments": segments,
        "ocr": ocr_metadata,
    }

    document_id = db.save_document(
        response,
        db.SCANNED_DOCUMENTS_COLLECTION if used_ocr else db.MODEL_ANSWERS_COLLECTION,
    )
    response["document_id"] = document_id

    return JSONResponse(response)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    response = {
        "status": (
            "ok"
            if OCR_ENGINE is not None
            else "loading"
        ),
        "device": OCR_CFG.device,
        "dtype": OCR_CFG.dtype,
    }

    if OCR_CFG.device == "cuda":
        device_index = torch.cuda.current_device()

        response.update(
            {
                "gpu": torch.cuda.get_device_name(
                    device_index
                ),
                "cuda_version": torch.version.cuda,
                "compute_capability": ".".join(
                    map(
                        str,
                        torch.cuda.get_device_capability(
                            device_index
                        ),
                    )
                ),
                "gpu_memory_allocated_gb": round(
                    torch.cuda.memory_allocated(
                        device_index
                    )
                    / (1024 ** 3),
                    2,
                ),
                "gpu_memory_reserved_gb": round(
                    torch.cuda.memory_reserved(
                        device_index
                    )
                    / (1024 ** 3),
                    2,
                ),
            }
        )

    return response