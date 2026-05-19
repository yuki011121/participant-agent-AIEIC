"""
Participant Agent for AIEIC Lab
Tracks student interactions and provides context for Lab Companion

Author: Yayun
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import uuid
import os
import logging
from dotenv import load_dotenv

from src.participant_agent import ParticipantAgent

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    logger.info("Participant Agent starting...")
    yield
    logger.info("Participant Agent shutting down...")


app = FastAPI(
    title="AIEIC Participant Agent",
    description="Tracks student interactions for personalized tutoring",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = ParticipantAgent()


# ============ Request/Response Models ============

class LogInteractionRequest(BaseModel):
    student_id: str
    session_id: str
    message: str
    response_time_ms: Optional[int] = None


class LogInteractionResponse(BaseModel):
    status: str
    interaction_id: str


class StudentContextResponse(BaseModel):
    total_questions: int
    question_type_distribution: dict
    avg_hint_level: float
    sessions_count: int
    avg_questions_per_session: float
    session_help_frequency: dict
    summary: str


# ---- Contract-required error schema ----

class ErrorDetail(BaseModel):
    code: str
    message: str
    agent: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def _make_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message,
                           "agent": "participant", "request_id": str(uuid.uuid4())}}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    code_map = {400: "BAD_REQUEST", 404: "NOT_FOUND", 422: "INVALID_REQUEST", 500: "INTERNAL_ERROR"}
    return _make_error(exc.status_code, code_map.get(exc.status_code, "HTTP_ERROR"), str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return _make_error(422, "INVALID_REQUEST", str(exc))


@app.exception_handler(Exception)
async def generic_exception_handler(_request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return _make_error(500, "INTERNAL_ERROR", "An internal error occurred")


# ---- New endpoints: StudentStatus ----

class StudentStatus(str, Enum):
    ON_TRACK = "on_track"
    NEEDS_HELP = "needs_help"
    FLAGGED = "flagged"
    INACTIVE = "inactive"


class StudentStatusResponse(BaseModel):
    student_id: str
    status: StudentStatus


class StudentSummary(BaseModel):
    student_id: str
    total_questions: int
    top_topic: str
    avg_hint_level: float
    status: StudentStatus
    last_message: str


class CohortResponse(BaseModel):
    lab_id: str
    students: list[StudentSummary]
    total_prompts: int
    avg_per_student: float
    hint_distribution: dict


# ============ Endpoints ============

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "ok", "agent": "participant", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint for Azure Container Apps"""
    return {"status": "healthy", "agent": "participant", "version": "0.1.0"}


@app.post("/participant/log", response_model=LogInteractionResponse)
async def log_interaction(request: LogInteractionRequest):
    """
    Log a student interaction
    Called by Lab Companion after every message is processed
    """
    try:
        logger.info(f"Logging interaction for student: {request.student_id}")
        interaction_id = await agent.log_interaction(
            student_id=request.student_id,
            session_id=request.session_id,
            message=request.message,
            response_time_ms=request.response_time_ms
        )
        logger.info(f"Interaction logged: {interaction_id}")
        return LogInteractionResponse(status="ok", interaction_id=interaction_id)
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/participant/context/{student_id}", response_model=StudentContextResponse)
async def get_student_context(student_id: str):
    """
    Get student context for personalization
    Called by Lab Companion at session start
    """
    try:
        context = await agent.get_student_context(student_id)
        return StudentContextResponse(**context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/participant/student/{student_id}/status", response_model=StudentStatusResponse)
async def get_student_status(student_id: str):
    """
    Get StudentStatus classification for a single student.
    Drives the Student Activity tab in the instructor dashboard.
    """
    try:
        status = await agent.get_student_status(student_id)
        return StudentStatusResponse(student_id=student_id, status=StudentStatus(status))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/participant/cohort/{lab_id}", response_model=CohortResponse)
async def get_cohort_context(lab_id: str):
    """
    Batch endpoint: per-student summaries + class-level aggregates for a lab.
    Replaces N per-student calls from the Orchestrator for the instructor dashboard.
    """
    try:
        result = await agent.get_cohort_context(lab_id)
        students = [
            StudentSummary(**{**s, "status": StudentStatus(s["status"])})
            for s in result["students"]
        ]
        return CohortResponse(
            lab_id=result["lab_id"],
            students=students,
            total_prompts=result["total_prompts"],
            avg_per_student=result["avg_per_student"],
            hint_distribution=result["hint_distribution"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
