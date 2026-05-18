"""
Participant Agent for AIEIC Lab
Tracks student interactions and provides context for Lab Companion

Author: Yayun
"""

from contextlib import asynccontextmanager
import uuid as uuid_module

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import os
import logging
from dotenv import load_dotenv

from src.participant_agent import ParticipantAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"
AGENT_NAME = "participant"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Participant Agent starting...")
    yield
    logger.info("Participant Agent shutting down...")


app = FastAPI(
    title="AIEIC Participant Agent",
    description="Tracks student interactions for personalized tutoring",
    version=APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ParticipantAgent()


# ============ Shared Error Schema (INTERFACE_CONTRACT §Shared Schemas) ============

class ErrorDetail(BaseModel):
    code: str
    message: str
    agent: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "agent": AGENT_NAME,
                "request_id": str(uuid_module.uuid4()),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    # Callers may raise with a dict detail carrying a specific code
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return _error_response(exc.status_code, exc.detail["code"], exc.detail["message"])
    _code_map = {400: "BAD_REQUEST", 404: "NOT_FOUND", 422: "INVALID_REQUEST", 500: "INTERNAL_ERROR"}
    code = _code_map.get(exc.status_code, "HTTP_ERROR")
    return _error_response(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(422, "INVALID_REQUEST", str(exc))


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return _error_response(500, "INTERNAL_ERROR", "An internal error occurred")


# ============ Enums ============

class StudentStatus(str, Enum):
    ON_TRACK = "on_track"
    NEEDS_HELP = "needs_help"
    FLAGGED = "flagged"
    INACTIVE = "inactive"


# ============ Request / Response Models ============

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
    return {"status": "ok", "agent": AGENT_NAME, "version": APP_VERSION}


@app.get("/health")
async def health_check():
    """Health check — shape required by INTERFACE_CONTRACT §Required Endpoints."""
    return {"status": "healthy", "agent": AGENT_NAME, "version": APP_VERSION}


@app.post("/participant/log", response_model=LogInteractionResponse)
async def log_interaction(request: LogInteractionRequest):
    """Log a student interaction. Called by Orchestrator after every student message."""
    logger.info(f"Logging interaction for student: {request.student_id}")
    interaction_id = await agent.log_interaction(
        student_id=request.student_id,
        session_id=request.session_id,
        message=request.message,
        response_time_ms=request.response_time_ms,
    )
    logger.info(f"Interaction logged: {interaction_id}")
    return LogInteractionResponse(status="ok", interaction_id=interaction_id)


@app.get("/participant/context/{student_id}", response_model=StudentContextResponse)
async def get_student_context(student_id: str):
    """
    Aggregated learning profile for one student.
    Called by Orchestrator at session start and refreshed periodically.
    """
    context = await agent.get_student_context(student_id)
    return StudentContextResponse(**context)


@app.get("/participant/student/{student_id}/status", response_model=StudentStatusResponse)
async def get_student_status(student_id: str):
    """
    Simple StudentStatus classification for one student.
    Drives the Student Activity tab UI in the instructor dashboard.
    """
    status = await agent.get_student_status(student_id)
    return StudentStatusResponse(student_id=student_id, status=StudentStatus(status))


@app.get("/participant/cohort/{lab_id}", response_model=CohortResponse)
async def get_cohort_context(lab_id: str):
    """
    Batch endpoint: per-student summaries + class-level aggregates for a lab.
    Primary data source for the Orchestrator instructor dashboard (replaces N per-student calls).
    """
    result = await agent.get_cohort_context(lab_id)
    # Coerce status strings to enum values
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


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
