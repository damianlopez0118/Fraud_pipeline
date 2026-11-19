"""Serve the local dashboard and its read-only chat API."""

import argparse
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Fraud Pipeline Dashboard", docs_url=None, redoc_url=None)
GREETING_MESSAGES = {
    "hi",
    "hi there",
    "hello",
    "hello there",
    "hey",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
}
GREETING_ANSWER = (
    "Hello! This is Gianna your fraud expert. How can I assist you today with the "
    "fraud-pipeline report?"
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatTurn]


def get_artifacts_dir() -> Path:
    return Path(os.getenv("FRAUD_PIPELINE_ARTIFACTS", "artifacts")).resolve()


def is_greeting(message: str) -> bool:
    return message.strip().casefold().rstrip("!.,?") in GREETING_MESSAGES


@lru_cache(maxsize=1)
def get_agent(artifacts_path: str, model_name: str, base_url: str):
    from .agent import build_agent

    return build_agent(Path(artifacts_path), model_name=model_name, base_url=base_url)


@app.get("/", response_class=FileResponse)
def dashboard():
    path = get_artifacts_dir() / "dashboard.html"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dashboard not found at {path}. Run fraud-pipeline first.",
        )
    return FileResponse(path, media_type="text/html")


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="End the conversation with a user message.")
    if len(request.messages) > 30 or any(len(turn.content) > 4000 for turn in request.messages):
        raise HTTPException(status_code=413, detail="Conversation is too long.")
    if len(request.messages) == 1 and is_greeting(request.messages[0].content):
        return {"answer": GREETING_ANSWER}

    from .agent import socure_update_answer

    update_answer = socure_update_answer(request.messages[-1].content, get_artifacts_dir())
    if update_answer is not None:
        return {"answer": update_answer}

    from langchain_core.messages import AIMessage, HumanMessage

    messages = [
        HumanMessage(content=turn.content)
        if turn.role == "user"
        else AIMessage(content=turn.content)
        for turn in request.messages
    ]
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        graph = get_agent(str(get_artifacts_dir()), model_name, base_url)
        result = graph.invoke({"messages": messages}, {"recursion_limit": 12})
    except Exception as error:
        logger.exception("Local fraud assistant request failed")
        raise HTTPException(
            status_code=503,
            detail="Local agent unavailable. Start Ollama and ensure the configured model is installed.",
        ) from error

    content = result["messages"][-1].content
    if isinstance(content, list):
        answer = "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        answer = str(content)
    return {"answer": answer}


@app.get("/socure", response_class=FileResponse)
def socure_dashboard():
    path = get_artifacts_dir() / "socure.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Socure snapshot unavailable. Run socure_refresh.")
    return FileResponse(path, media_type="text/html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local fraud dashboard and assistant")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["FRAUD_PIPELINE_ARTIFACTS"] = str(args.artifacts.resolve())

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)