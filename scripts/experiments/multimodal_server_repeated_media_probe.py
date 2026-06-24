# SPDX-License-Identifier: Apache-2.0
"""Probe repeated-media serving latency and encoder-cache behavior.

The script drives the OpenAI-compatible ``/v1/chat/completions`` endpoint with
the same local media item multiple times. It is intentionally small so it can
be used against Qwen3-Omni and Ming-Omni servers without dataset downloads.
Server logs remain the source of truth for model-internal cache traces; this
client records request timing, usage, status, and response snippets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


@dataclasses.dataclass
class RequestTiming:
    phase: str
    request_id: str
    index: int
    media: str
    status_code: int
    ok: bool
    latency_seconds: float
    first_event_seconds: float | None = None
    first_delta_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_text: str = ""
    error: str | None = None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def _summarize_phase(items: list[RequestTiming]) -> dict[str, Any]:
    ok_items = [item for item in items if item.ok]
    latencies = [item.latency_seconds for item in ok_items]
    deltas = [
        item.first_delta_seconds
        for item in ok_items
        if item.first_delta_seconds is not None
    ]
    return {
        "requests": len(items),
        "successful": len(ok_items),
        "failed": len(items) - len(ok_items),
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "first_delta_seconds": {
            "mean": statistics.fmean(deltas) if deltas else None,
            "min": min(deltas) if deltas else None,
            "p50": _percentile(deltas, 0.50),
            "p95": _percentile(deltas, 0.95),
            "max": max(deltas) if deltas else None,
        },
    }


def _media_field(media_kind: str) -> str:
    if media_kind == "image":
        return "images"
    if media_kind == "audio":
        return "audios"
    if media_kind == "video":
        return "videos"
    raise ValueError(f"unsupported media kind: {media_kind}")


def _extract_text_from_nonstream(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_usage(body: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )


def _send_one(args: argparse.Namespace, phase: str, index: int, media: str) -> RequestTiming:
    request_id = f"{args.request_id_prefix}-{phase}-{index}"
    payload: dict[str, Any] = {
        "model": args.model,
        "request_id": request_id,
        "messages": [{"role": "user", "content": args.prompt}],
        _media_field(args.media_kind): [media],
        "modalities": ["text"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
        "seed": args.seed + index,
    }
    if args.media_kind == "video":
        if args.video_fps is not None:
            payload["video_fps"] = args.video_fps
        if args.video_max_frames is not None:
            payload["video_max_frames"] = args.video_max_frames
        if args.video_max_pixels is not None:
            payload["video_max_pixels"] = args.video_max_pixels

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    start = time.perf_counter()
    first_event: float | None = None
    first_delta: float | None = None
    response_text = ""
    status_code = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    try:
        timeout = httpx.Timeout(args.timeout_s, connect=min(args.timeout_s, 30.0))
        with httpx.Client(timeout=timeout) as client:
            if args.stream:
                with client.stream("POST", url, json=payload) as response:
                    status_code = response.status_code
                    if status_code >= 400:
                        error_body = response.read().decode("utf-8", errors="replace")
                        raise RuntimeError(f"HTTP {status_code}: {error_body[:512]}")
                    for raw_line in response.iter_lines():
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        body_text = line[len("data:") :].strip()
                        if body_text == "[DONE]":
                            continue
                        now = time.perf_counter()
                        if first_event is None:
                            first_event = now - start
                        try:
                            event = json.loads(body_text)
                        except json.JSONDecodeError:
                            continue
                        prompt_tokens, completion_tokens = _extract_usage(event)
                        for choice in event.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                response_text += content
                                if first_delta is None:
                                    first_delta = now - start
                            audio = delta.get("audio") or {}
                            if audio.get("data") and first_delta is None:
                                first_delta = now - start
            else:
                response = client.post(url, json=payload)
                status_code = response.status_code
                if status_code >= 400:
                    raise RuntimeError(f"HTTP {status_code}: {response.text[:512]}")
                body = response.json()
                response_text = _extract_text_from_nonstream(body)
                prompt_tokens, completion_tokens = _extract_usage(body)
    except Exception as exc:  # noqa: BLE001 - benchmark records failures in JSON.
        return RequestTiming(
            phase=phase,
            request_id=request_id,
            index=index,
            media=media,
            status_code=status_code,
            ok=False,
            latency_seconds=time.perf_counter() - start,
            first_event_seconds=first_event,
            first_delta_seconds=first_delta,
            response_text=response_text[: args.response_chars],
            error=str(exc),
        )

    return RequestTiming(
        phase=phase,
        request_id=request_id,
        index=index,
        media=media,
        status_code=status_code,
        ok=True,
        latency_seconds=time.perf_counter() - start,
        first_event_seconds=first_event,
        first_delta_seconds=first_delta,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_text=response_text[: args.response_chars],
    )


def _run_concurrent_phase(
    args: argparse.Namespace,
    *,
    phase: str,
    count: int,
    media: str,
) -> list[RequestTiming]:
    if count <= 0:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(_send_one, args, phase, index, media)
            for index in range(count)
        ]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def _write_markdown(output: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Multimodal Repeated-Media Server Probe",
        "",
        f"- model: `{data['config']['model']}`",
        f"- base_url: `{data['config']['base_url']}`",
        f"- media_kind: `{data['config']['media_kind']}`",
        f"- primary_media: `{data['config']['primary_media']}`",
        f"- stream: `{data['config']['stream']}`",
        "",
        "## Summary",
        "",
        "| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, summary in data["summary"].items():
        lat = summary["latency_seconds"]
        delta = summary["first_delta_seconds"]
        lines.append(
            "| {phase} | {requests} | {ok} | {mean} | {p50} | {p95} | {delta_mean} |".format(
                phase=phase,
                requests=summary["requests"],
                ok=summary["successful"],
                mean=_fmt(lat["mean"]),
                p50=_fmt(lat["p50"]),
                p95=_fmt(lat["p95"]),
                delta_mean=_fmt(delta["mean"]),
            )
        )
    lines.extend(["", "## Requests", ""])
    lines.append(
        "| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in data["requests"]:
        lines.append(
            "| {phase} | `{request_id}` | {status} | {latency} | {delta} | {ptok} | {ctok} | {error} |".format(
                phase=row["phase"],
                request_id=row["request_id"],
                status=row["status_code"],
                latency=_fmt(row["latency_seconds"]),
                delta=_fmt(row["first_delta_seconds"]),
                ptok=row["prompt_tokens"] if row["prompt_tokens"] is not None else "",
                ctok=(
                    row["completion_tokens"]
                    if row["completion_tokens"] is not None
                    else ""
                ),
                error=(row["error"] or "").replace("|", "\\|"),
            )
        )
    output.write_text("\n".join(lines) + "\n")


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _resolve_media(value: str | None) -> str | None:
    if value is None:
        return None
    if "://" in value:
        return value
    return str(Path(value).expanduser().resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated local media requests against an omni server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--media-kind", choices=["image", "audio", "video"], default="image")
    parser.add_argument("--media", required=True, help="Primary media path or URL.")
    parser.add_argument("--alt-media", help="Optional different media path or URL.")
    parser.add_argument("--prompt", default="Describe this image in one short sentence.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--request-id-prefix", default=f"mm-cache-{int(time.time())}")
    parser.add_argument("--cold-concurrency", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--response-chars", type=int, default=240)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--video-max-frames", type=int)
    parser.add_argument("--video-max-pixels", type=int)
    args = parser.parse_args()

    primary_media = _resolve_media(args.media)
    alt_media = _resolve_media(args.alt_media)
    assert primary_media is not None

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["primary_media"] = primary_media
    config["alt_media"] = alt_media

    requests: list[RequestTiming] = []
    requests.extend(
        _run_concurrent_phase(
            args,
            phase="cold_concurrent_same_media",
            count=args.cold_concurrency,
            media=primary_media,
        )
    )

    for index in range(args.repeats):
        requests.append(_send_one(args, "warm_sequential_same_media", index, primary_media))

    if alt_media:
        requests.append(_send_one(args, "single_different_media", 0, alt_media))

    requests.sort(key=lambda item: (item.phase, item.index, item.request_id))
    requests_by_phase: dict[str, list[RequestTiming]] = {}
    for item in requests:
        requests_by_phase.setdefault(item.phase, []).append(item)

    result = {
        "config": config,
        "summary": {
            phase: _summarize_phase(items) for phase, items in requests_by_phase.items()
        },
        "requests": [dataclasses.asdict(item) for item in requests],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.markdown_output, result)

    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if any(not item.ok for item in requests):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
