import json
import logging
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any

from ..core.config import UPSTREAM_URL, ENABLE_ATTENTION_FILTER
from ..core.filter import WAMPruner

logger = logging.getLogger(__name__)

app = FastAPI(title="WAMP - OpenAI API Proxy with Attention Filter")

filter_pruner = None


@app.on_event("startup")
async def startup_event():
    global filter_pruner
    if ENABLE_ATTENTION_FILTER:
        try:
            filter_pruner = WAMPruner()
            logger.info("✓ Attention filter initialized")
        except Exception as e:
            logger.error(f"⚠ Failed to initialize filter: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "wamp",
        "upstream": UPSTREAM_URL,
        "filter_enabled": filter_pruner is not None,
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_catch_all(path: str, request: Request):
    if path == "health":
        return await health()
    return await proxy_request(path, request)


async def apply_filter(body_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply attention filter to messages"""
    if not filter_pruner or "messages" not in body_dict:
        return body_dict

    messages = body_dict.get("messages", [])
    if not messages:
        return body_dict

    # Get the last user message as the core task
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    user_query = filter_pruner.get_content(last_user).strip() if last_user else ""

    if not user_query:
        return body_dict

    # Enhanced task prompt to guide attention
    task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{user_query}'"

    try:
        filtered = filter_pruner.get_attention_filtered(messages, task)

        body_dict["messages"] = filtered
        stype = "stream" if body_dict.get("stream") else "non-stream"
        logger.info(f"[Filter] {stype}: {len(messages)} → {len(filtered)} msgs")
    except Exception as e:
        logger.error(f"Filter error: {e}")

    return body_dict


async def proxy_request(path: str, request: Request):
    url = f"{UPSTREAM_URL}/{path}"
    body = await request.body()
    body_dict = None

    if body:
        try:
            body_dict = json.loads(body)
        except json.JSONDecodeError:
            pass

    if body_dict and "messages" in body_dict:
        body_dict = await apply_filter(body_dict)
        body = json.dumps(body_dict).encode()

    is_stream = body_dict.get("stream") if isinstance(body_dict, dict) else False
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # Forward the request to upstream
    if is_stream:
        return await stream_proxy(url, headers, body)
    else:
        return await simple_proxy(request.method, url, headers, body)


async def stream_proxy(url: str, headers: dict, body: bytes):
    async def event_generator():
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST", url, headers=headers, content=body, timeout=300.0
                ) as resp:
                    async for chunk in resp.aiter_text():
                        if chunk:
                            yield chunk
            except Exception as e:
                logger.error(f"Streaming proxy error: {e}")
                yield f"data: {json.dumps({'error': {'type': 'proxy_error', 'message': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def simple_proxy(method: str, url: str, headers: dict, body: bytes):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.request(method, url, headers=headers, content=body, timeout=300.0)
            ct = resp.headers.get("content-type", "")
            if "application/json" in ct:
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            return JSONResponse(status_code=resp.status_code, content={"text": resp.text})
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream timeout")
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
