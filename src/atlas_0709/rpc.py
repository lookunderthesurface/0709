from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

from .target_runtime import DirectFlashInferMaskedTreeVerifyBackend, verify_payload_from_mapping
from .types import TargetRouteScore, TargetVerifyResult


def target_score_to_dict(score: TargetRouteScore) -> dict[str, object]:
    return {
        "route_id": int(score.route_id),
        "token_ids": [int(token_id) for token_id in score.token_ids],
        "target_logprob": float(score.target_logprob),
        "draft_logprob": float(score.draft_logprob),
        "token_logprobs": [float(value) for value in score.token_logprobs],
        "first_token_logprob": (
            None if score.first_token_logprob is None else float(score.first_token_logprob)
        ),
        "selection_score": (
            None if score.selection_score is None else float(score.selection_score)
        ),
        "score_weights": [float(value) for value in score.score_weights],
    }


def target_verify_result_to_dict(result: TargetVerifyResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "decision": str(result.decision),
        "scores": [target_score_to_dict(score) for score in result.scores],
        "metadata": dict(result.metadata),
    }
    if result.decision == "fallback_ar":
        payload.update(
            {
                "selected_route_id": None,
                "fallback_token_ids": [int(token_id) for token_id in result.fallback_token_ids],
                "fallback_reason": result.fallback_reason,
            }
        )
        return payload
    payload.update(
        {
            "selected_route_id": int(result.selected_route_id),
            "selected_score": target_score_to_dict(result.selected_score),
        }
    )
    return payload


@dataclass
class TargetServerApp:
    backend: DirectFlashInferMaskedTreeVerifyBackend

    def __post_init__(self) -> None:
        self.lock = threading.Lock()

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "service": "atlas_0709_target",
            "metadata": self.backend.runtime_metadata(),
        }

    def prefill(self, request: Mapping[str, Any]) -> dict[str, object]:
        token_ids = request.get("prompt_token_ids")
        if not isinstance(token_ids, list):
            raise ValueError("prefill requires prompt_token_ids: list[int]")
        with self.lock:
            prefix = self.backend.prefill([int(token_id) for token_id in token_ids])
        return {
            "ok": True,
            "prompt_len": int(prefix.committed_length),
            "metadata": dict(prefix.metadata),
        }

    def verify(self, request: Mapping[str, Any]) -> dict[str, object]:
        prefix = request.get("prefix_token_ids")
        routes = request.get("routes")
        fallback_max_tokens = request.get("fallback_max_tokens")
        eos_token_id = request.get("eos_token_id")
        if not isinstance(prefix, list):
            raise ValueError("verify requires prefix_token_ids: list[int]")
        if not isinstance(routes, list):
            raise ValueError("verify requires routes: list[object]")
        payloads = [verify_payload_from_mapping(dict(item)) for item in routes]
        with self.lock:
            result = self.backend.verify_payloads(
                prefix_token_ids=[int(token_id) for token_id in prefix],
                routes=payloads,
                fallback_max_tokens=(
                    None if fallback_max_tokens is None else int(fallback_max_tokens)
                ),
                eos_token_id=(None if eos_token_id is None else int(eos_token_id)),
            )
        return {
            "ok": True,
            **target_verify_result_to_dict(result),
        }


@dataclass
class InProcessTargetClient:
    """Target client compatible with the generators, without HTTP or threads."""

    backend: DirectFlashInferMaskedTreeVerifyBackend

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "service": "atlas_0709_target_in_process",
            "transport": "in_process",
            "metadata": self.backend.runtime_metadata(),
        }

    def prefill(self, prompt_token_ids: Sequence[int]) -> dict[str, object]:
        prefix = self.backend.prefill(prompt_token_ids)
        return {
            "ok": True,
            "prompt_len": int(prefix.committed_length),
            "metadata": dict(prefix.metadata),
        }

    def verify(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[Mapping[str, object]],
        fallback_max_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> dict[str, object]:
        result = self.backend.verify_payloads(
            prefix_token_ids=prefix_token_ids,
            routes=[verify_payload_from_mapping(dict(item)) for item in routes],
            fallback_max_tokens=fallback_max_tokens,
            eos_token_id=eos_token_id,
        )
        return {"ok": True, **target_verify_result_to_dict(result)}


def make_target_handler(app: TargetServerApp) -> type[BaseHTTPRequestHandler]:
    class TargetRPCHandler(BaseHTTPRequestHandler):
        server_version = "Atlas0709TargetRPC/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(200, app.health())
                return
            self._write_json(404, {"ok": False, "error": f"unknown endpoint: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                if self.path == "/prefill":
                    response = app.prefill(payload)
                elif self.path == "/verify":
                    response = app.verify(payload)
                else:
                    self._write_json(404, {"ok": False, "error": f"unknown endpoint: {self.path}"})
                    return
                self._write_json(200, response)
            except Exception as exc:
                self._write_json(500, {"ok": False, "error": repr(exc)})

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[target-rpc] {self.address_string()} - {fmt % args}", flush=True)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return TargetRPCHandler


def serve_target(app: TargetServerApp, *, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, int(port)), make_target_handler(app))
    print(
        json.dumps(
            {
                "service": "atlas_0709_target",
                "host": host,
                "port": int(port),
                "metadata": app.backend.runtime_metadata(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class RemoteTargetClient:
    def __init__(self, base_url: str, *, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def health(self) -> dict[str, object]:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def prefill(self, prompt_token_ids: Sequence[int]) -> dict[str, object]:
        return self._post(
            "/prefill",
            {"prompt_token_ids": [int(token_id) for token_id in prompt_token_ids]},
        )

    def verify(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[Mapping[str, object]],
        fallback_max_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "prefix_token_ids": [int(token_id) for token_id in prefix_token_ids],
            "routes": [dict(route) for route in routes],
        }
        if fallback_max_tokens is not None:
            payload["fallback_max_tokens"] = int(fallback_max_tokens)
        if eos_token_id is not None:
            payload["eos_token_id"] = int(eos_token_id)
        return self._post("/verify", payload)

    def _post(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"content-type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"target RPC {path} failed with HTTP {exc.code}: {body}") from exc
        if not result.get("ok"):
            raise RuntimeError(f"target RPC {path} failed: {result}")
        return result


def selected_route_id_from_response(response: Mapping[str, object]) -> int:
    if str(response.get("decision", "select")) != "select":
        raise ValueError(f"target response did not select a route: {response.get('decision')}")
    return int(response["selected_route_id"])
