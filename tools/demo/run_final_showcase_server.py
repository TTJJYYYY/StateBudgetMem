#!/usr/bin/env python3
"""Serve the final showcase with an optional DeepSeek answer endpoint.

The static ``index.html`` remains the default artifact. This server is only for
the defense demo path where a user types a DeepSeek API key into the browser.
The key is accepted for a single request, forwarded to DeepSeek, and never
written to disk or logs.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHOWCASE_DIR = ROOT / "results" / "final_showcase"
DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve final_showcase and proxy demo-only DeepSeek answers.",
    )
    parser.add_argument("--showcase-dir", type=Path, default=DEFAULT_SHOWCASE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--deepseek-url", default=DEFAULT_DEEPSEEK_URL)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    showcase_dir = args.showcase_dir.resolve()
    if not (showcase_dir / "index.html").exists():
        raise SystemExit(
            f"index.html not found in {showcase_dir}; run build_final_showcase.py first"
        )

    handler = make_handler(
        showcase_dir=showcase_dir,
        deepseek_url=args.deepseek_url,
        timeout_s=args.timeout_s,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/index.html"
    print(f"Serving StateBudgetMem final_showcase at {url}")
    print("DeepSeek demo endpoint: /api/free-question-answer")
    print("API keys are accepted per request and are not saved.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
    return 0


def make_handler(
    *,
    showcase_dir: Path,
    deepseek_url: str,
    timeout_s: float,
) -> type[SimpleHTTPRequestHandler]:
    class FinalShowcaseHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(showcase_dir), **kwargs)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/api/free-question-answer":
                self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
                return
            try:
                payload = self._read_json_body()
                response = build_deepseek_answers(
                    payload,
                    deepseek_url=deepseek_url,
                    timeout_s=timeout_s,
                )
                self._write_json(HTTPStatus.OK, response)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                self._write_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": f"DeepSeek HTTP {exc.code}: {detail[:500]}"},
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                self._write_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": f"DeepSeek request failed: {exc}"},
                )

        def log_message(self, format: str, *args: Any) -> None:
            # Keep the default concise log but never include request bodies.
            super().log_message(format, *args)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("empty request body")
            if length > 200_000:
                raise ValueError("request body too large for demo endpoint")
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return FinalShowcaseHandler


def build_deepseek_answers(
    payload: dict[str, Any],
    *,
    deepseek_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    api_key = str(payload.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("missing DeepSeek API key")
    model = str(payload.get("model") or "deepseek-chat").strip()
    query = str(payload.get("query") or "").strip()
    columns = payload.get("columns")
    if not query:
        raise ValueError("missing query")
    if not isinstance(columns, list) or not columns:
        raise ValueError("columns must be a non-empty list")

    started = time.perf_counter()
    answers: dict[str, Any] = {}
    for column in columns:
        if not isinstance(column, dict):
            raise ValueError("each column must be an object")
        column_id = str(column.get("column_id") or "").strip()
        if not column_id:
            raise ValueError("column_id is required")
        answer = call_deepseek_for_column(
            api_key=api_key,
            model=model,
            query=query,
            column=column,
            deepseek_url=deepseek_url,
            timeout_s=timeout_s,
        )
        answers[column_id] = answer

    return {
        "answerer": "deepseek_api_demo",
        "provider": "deepseek",
        "model": model,
        "answers": answers,
        "total_latency_ms": (time.perf_counter() - started) * 1000.0,
        "boundary": (
            "Demo-only generated answers. Not used for formal metrics; formal "
            "conclusions come from results/fair_comparison_v2."
        ),
    }


def call_deepseek_for_column(
    *,
    api_key: str,
    model: str,
    query: str,
    column: dict[str, Any],
    deepseek_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    request_body = {
        "model": model,
        "temperature": 0.2,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 StateBudgetMem 答辩展示中的本地代理回答组织器。"
                    "只根据用户问题和给定 memory context 作答；不要编造新记忆。"
                    "如果没有 memory context，要明确说明无法确认个人状态。"
                    "回答要简洁，先给结论，再说明引用了哪些 memory ids。"
                    "这是 demo-only，不用于正式实验指标。"
                ),
            },
            {
                "role": "user",
                "content": build_column_prompt(query, column),
            },
        ],
    }
    body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        deepseek_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    latency_ms = (time.perf_counter() - started) * 1000.0
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("DeepSeek returned no choices")
    message = choices[0].get("message") or {}
    answer_text = str(message.get("content") or "").strip()
    if not answer_text:
        raise ValueError("DeepSeek returned an empty answer")
    usage = data.get("usage") or {}
    return {
        "answer_text": answer_text,
        "latency_ms": latency_ms,
        "usage": usage,
    }


def build_column_prompt(query: str, column: dict[str, Any]) -> str:
    memories = column.get("memory_context") or []
    if memories:
        context_lines = []
        for memory in memories:
            if not isinstance(memory, dict):
                continue
            context_lines.append(
                "- "
                f"id={memory.get('memory_id')}; "
                f"status={memory.get('status')}; "
                f"operation={memory.get('operation')}; "
                f"valid={memory.get('valid_from')} to {memory.get('valid_to')}; "
                f"text={memory.get('text')}"
            )
        context = "\n".join(context_lines)
    else:
        context = "(no memory context)"
    return (
        f"用户问题：{query}\n"
        f"展示列：{column.get('title')}\n"
        f"query type heuristic：{column.get('query_type_heuristic')}\n"
        f"heuristic reason：{column.get('query_type_reason')}\n"
        f"retrieved memory ids：{column.get('retrieved_memory_ids')}\n"
        f"memory context：\n{context}\n\n"
        "请基于本列上下文组织回答。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
