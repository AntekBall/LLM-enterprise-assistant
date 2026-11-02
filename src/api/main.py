from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.inference import load_llm
from src.models.rag import answer, get_reranker

DEPARTMENTS = [
    "engineering",
    "finance",
    "hiring",
    "legal",
    "people-group",
    "people-policies",
    "security",
    "support",
    "total-rewards",
]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    departments: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    n: int
    source: str
    title: str
    section: str
    department: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_llm()
    get_reranker()
    yield


app = FastAPI(title="Enterprise Handbook Assistant", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/departments")
def departments() -> list[str]:
    return DEPARTMENTS


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    unknown = set(req.departments or []) - set(DEPARTMENTS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown departments: {sorted(unknown)}")
    return answer(req.question, req.departments, req.top_k)
