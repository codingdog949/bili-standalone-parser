from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import sys
import tempfile
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

# Minimal standalone parser:
# - Input: a Bilibili URL
# - Output: stable JSON on stdout
# - Diagnostics: structured logs to file + machine-readable error JSON on stderr
_BVID_PATTERN = re.compile(r"BV[0-9A-Za-z]{10}")
_PLAYINFO_PATTERN = re.compile(r"__playinfo__\s*=\s*(\{.*?\})\s*</script>", re.S)
_SHORT_LINK_HOSTS = {"b23.tv", "www.b23.tv"}
_CANONICAL_HOSTS = {"www.bilibili.com", "bilibili.com"}
_MODEL_ALIASES = {"whisper-small": "small", "whisper-base": "base", "whisper-tiny": "tiny"}
_COREML_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx-fp32",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-turbo",
}
logger = logging.getLogger("min_bili_parse")


class ErrorCode:
    INVALID_URL = "INVALID_URL"
    VIDEO_NOT_FOUND = "VIDEO_NOT_FOUND"
    AUDIO_FETCH_FAILED = "AUDIO_FETCH_FAILED"
    ASR_FAILED = "ASR_FAILED"
    UPSTREAM_RATE_LIMIT = "UPSTREAM_RATE_LIMIT"
    UPSTREAM_CONNECT_FAILED = "UPSTREAM_CONNECT_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ParserError(Exception):
    def __init__(self, code: str, message: str, *, stage: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


class StandaloneBiliParser:
    def __init__(self, timeout_sec: float = 12.0, temp_dir: str = "/tmp", max_audio_mb: int = 200, cookie: str | None = None):
        headers = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
            "referer": "https://www.bilibili.com/",
        }
        if cookie:
            headers["cookie"] = cookie
        self._max_audio_bytes = max_audio_mb * 1024 * 1024
        self._temp_dir = temp_dir
        use_env_proxy = os.getenv("USE_ENV_PROXY", "1").lower() not in {"0", "false", "no"}
        self._client = httpx.AsyncClient(
            timeout=timeout_sec,
            follow_redirects=True,
            headers=headers,
            trust_env=use_env_proxy,
        )
        logger.info("http client ready timeout=%.1fs proxy_env=%s", timeout_sec, use_env_proxy)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def parse(self, url: str, asr_model: str = "whisper-small") -> dict[str, Any]:
        logger.info("parse start url=%s asr_model=%s", url, asr_model)
        canonical_url, bvid = await self._normalize_url(url)
        logger.info("normalized bvid=%s canonical=%s", bvid, canonical_url)
        video = await self._fetch_video_meta(bvid)
        logger.info("video meta title=%s cid=%s", video["title"], video["cid"])

        source = "subtitle"
        warnings: list[str] = []
        subtitle = await self._fetch_subtitle(video["bvid"], video["cid"])
        if subtitle:
            language, segments = subtitle
            logger.info("subtitle found language=%s segments=%d", language, len(segments))
        else:
            # No subtitle: switch to ASR fallback (network audio fetch + local transcription).
            source = "asr"
            warnings.append("subtitle unavailable; fallback to ASR")
            logger.warning("subtitle missing; fallback to ASR")
            audio_url = await self._fetch_audio_stream_url(canonical_url)
            logger.info("audio url resolved")
            audio_path = await self._download_audio(audio_url)
            logger.info("audio downloaded path=%s", audio_path)
            try:
                language, segments = await asyncio.to_thread(_transcribe_audio, audio_path, asr_model)
                logger.info("asr done language=%s segments=%d", language, len(segments))
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info("temp audio removed path=%s", audio_path)

        segments = _clean_and_merge(segments)
        full_text = "\n".join(s["text"] for s in segments if s["text"].strip())
        if not full_text.strip():
            raise ParserError(ErrorCode.INTERNAL_ERROR, "empty text after processing", stage="postprocess")

        logger.info("parse done source=%s text_len=%d", source, len(full_text))
        return {
            "bvid": video["bvid"],
            "title": video["title"],
            "source": source,
            "language": language,
            "full_text": full_text,
            "summary": _summary(full_text),
            "warnings": warnings,
        }

    async def _normalize_url(self, url: str) -> tuple[str, str]:
        text = _normalize_input_url(url)
        parsed = urlparse(text)
        if (parsed.hostname or "").lower() in _SHORT_LINK_HOSTS:
            # Resolve b23 short links but strictly enforce allowed final hosts.
            response = await self._client.get(text)
            self._raise_for_upstream(response)
            resolved = str(response.url)
            resolved_host = (urlparse(resolved).hostname or "").lower()
            if resolved_host not in _CANONICAL_HOSTS:
                raise ParserError(ErrorCode.INVALID_URL, "short url resolved to unsupported host", stage="normalize_url")
            text = resolved
        bvid = _extract_bvid(text)
        return f"https://www.bilibili.com/video/{bvid}", bvid

    async def _fetch_video_meta(self, bvid: str) -> dict[str, Any]:
        response = await self._client.get("https://api.bilibili.com/x/web-interface/view", params={"bvid": bvid})
        self._raise_for_upstream(response)
        payload = response.json()
        if payload.get("code") != 0 or not payload.get("data"):
            raise ParserError(ErrorCode.VIDEO_NOT_FOUND, f"video not found: {bvid}", stage="fetch_meta")
        data = payload["data"]
        cid = int(data.get("cid") or data.get("pages", [{}])[0].get("cid") or 0)
        if not cid:
            raise ParserError(ErrorCode.VIDEO_NOT_FOUND, "cid missing from video metadata", stage="fetch_meta")
        return {"bvid": bvid, "cid": cid, "title": data.get("title") or bvid}

    async def _fetch_subtitle(self, bvid: str, cid: int) -> tuple[str, list[dict[str, Any]]] | None:
        response = await self._client.get("https://api.bilibili.com/x/player/v2", params={"bvid": bvid, "cid": cid})
        self._raise_for_upstream(response)
        payload = response.json()
        if payload.get("code") != 0:
            return None
        subtitles = (((payload.get("data") or {}).get("subtitle") or {}).get("subtitles") or [])
        if not subtitles:
            return None
        track = _pick_subtitle_track(subtitles)
        if not track:
            return None
        subtitle_url = track.get("subtitle_url")
        if not subtitle_url:
            return None
        if subtitle_url.startswith("//"):
            subtitle_url = "https:" + subtitle_url
        sub_resp = await self._client.get(subtitle_url)
        self._raise_for_upstream(sub_resp)
        body = (sub_resp.json().get("body") or [])
        segments = [
            {
                "start": float(item.get("from", 0.0)),
                "end": float(item.get("to", 0.0)),
                "text": str(item.get("content", "")).strip(),
            }
            for item in body
            if str(item.get("content", "")).strip()
        ]
        if not segments:
            return None
        language = track.get("lan_doc") or track.get("lan") or "unknown"
        return language, segments

    async def _fetch_audio_stream_url(self, canonical_url: str) -> str:
        response = await self._client.get(canonical_url)
        self._raise_for_upstream(response)
        match = _PLAYINFO_PATTERN.search(response.text)
        if not match:
            raise ParserError(ErrorCode.AUDIO_FETCH_FAILED, "failed to parse playinfo from page", stage="fetch_audio")
        try:
            playinfo = json.loads(unescape(match.group(1)))
            audio_entries = (((playinfo.get("data") or {}).get("dash") or {}).get("audio") or [])
            if not audio_entries:
                raise ValueError("audio stream missing")
            return audio_entries[0]["baseUrl"]
        except Exception as exc:  # noqa: BLE001
            raise ParserError(ErrorCode.AUDIO_FETCH_FAILED, f"cannot decode playinfo: {exc}", stage="fetch_audio") from exc

    async def _download_audio(self, audio_url: str) -> str:
        with tempfile.NamedTemporaryFile(prefix="bili-audio-", suffix=".m4a", delete=False, dir=self._temp_dir) as fp:
            path = fp.name
        total = 0
        try:
            async with self._client.stream("GET", audio_url) as response:
                self._raise_for_upstream(response)
                with open(path, "wb") as out:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self._max_audio_bytes:
                            raise ParserError(
                                ErrorCode.AUDIO_FETCH_FAILED,
                                f"audio larger than max limit ({self._max_audio_bytes} bytes)",
                                stage="download_audio",
                            )
                        out.write(chunk)
        except Exception:
            if os.path.exists(path):
                os.remove(path)
            raise
        return path

    @staticmethod
    def _raise_for_upstream(response: httpx.Response) -> None:
        if response.status_code == 429:
            raise ParserError(ErrorCode.UPSTREAM_RATE_LIMIT, "upstream rate limited")
        if response.status_code >= 400:
            raise ParserError(ErrorCode.INTERNAL_ERROR, f"upstream request failed: {response.status_code}")


def _normalize_input_url(url: str) -> str:
    text = url.strip()
    if not text:
        raise ParserError(ErrorCode.INVALID_URL, "url is empty", stage="normalize_url")
    parsed = urlparse(text)
    if not parsed.scheme and parsed.path:
        return "https://" + text
    if parsed.scheme not in {"http", "https"}:
        raise ParserError(ErrorCode.INVALID_URL, "unsupported url scheme", stage="normalize_url")
    return text


def _extract_bvid(url: str) -> str:
    match = _BVID_PATTERN.search(url)
    if not match:
        raise ParserError(ErrorCode.INVALID_URL, "cannot find bvid in url", stage="normalize_url")
    return match.group(0)


def _pick_subtitle_track(subtitles: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred = ["zh-CN", "zh-Hans", "zh", "ai-zh"]
    for tag in preferred:
        for item in subtitles:
            if tag.lower() in (item.get("lan") or "").lower():
                return item
    return subtitles[0] if subtitles else None


def _clean_and_merge(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        text = _normalize_text(str(seg.get("text", "")))
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if out and start - out[-1]["end"] < 0.8 and len(text) < 50:
            out[-1]["end"] = max(float(out[-1]["end"]), end)
            out[-1]["text"] = f"{out[-1]['text']} {text}".strip()
        else:
            out.append({"start": start, "end": end, "text": text})
    return out


def _normalize_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\b(um+|uh+|啊+|呃+)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _summary(text: str) -> str | None:
    sentences = [s.strip() for s in re.split(r"[。！？!?\n]", text) if s.strip()]
    if not sentences:
        return None
    return "；".join(sentences[:3])


def _transcribe_audio_faster_whisper(audio_path: str, model: str) -> tuple[str, list[dict[str, Any]]]:
    resolved_model = _resolve_model_name(model)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ParserError(
            ErrorCode.ASR_FAILED,
            "faster-whisper not installed; install with `pip install faster-whisper`",
            stage="asr",
        ) from exc
    try:
        model_obj = WhisperModel(resolved_model, device="cpu", compute_type="int8")
        segments_iter, info = model_obj.transcribe(audio_path, vad_filter=True)
        segments = [
            {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
            for s in segments_iter
            if (s.text or "").strip()
        ]
        if not segments:
            raise ParserError(ErrorCode.ASR_FAILED, "asr returned empty transcript", stage="asr")
        language = getattr(info, "language", "unknown")
        return language, segments
    except ParserError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParserError(ErrorCode.ASR_FAILED, f"asr failed: {exc}", stage="asr") from exc


def _transcribe_audio_coreml(audio_path: str, model: str) -> tuple[str, list[dict[str, Any]]]:
    model_repo = _resolve_coreml_model_repo(model)
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ParserError(
            ErrorCode.ASR_FAILED,
            "mlx-whisper not installed; install with `pip install mlx-whisper`",
            stage="asr",
        ) from exc

    try:
        output = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_repo, word_timestamps=False)
        raw_segments = output.get("segments") or []
        segments = [
            {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": str(s.get("text", "")).strip()}
            for s in raw_segments
            if str(s.get("text", "")).strip()
        ]
        if not segments:
            text = str(output.get("text", "")).strip()
            if text:
                segments = [{"start": 0.0, "end": 0.0, "text": text}]
        if not segments:
            raise ParserError(ErrorCode.ASR_FAILED, "asr returned empty transcript", stage="asr")
        language = str(output.get("language", "unknown"))
        return language, segments
    except ParserError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParserError(ErrorCode.ASR_FAILED, f"asr failed: {exc}", stage="asr") from exc


def _transcribe_audio(audio_path: str, model: str) -> tuple[str, list[dict[str, Any]]]:
    backend = (os.getenv("ASR_BACKEND", "").strip().lower()) or _default_asr_backend()
    if backend in {"coreml", "mlx"}:
        try:
            return _transcribe_audio_coreml(audio_path, model)
        except ParserError as exc:
            if _is_enabled("ASR_COREML_FALLBACK_TO_FASTER_WHISPER", "1"):
                logger.warning(
                    "coreml backend failed (code=%s stage=%s message=%s); fallback to faster-whisper",
                    exc.code,
                    exc.stage,
                    exc.message,
                )
                return _transcribe_audio_faster_whisper(audio_path, model)
            raise
    if backend in {"faster-whisper", "cpu"}:
        return _transcribe_audio_faster_whisper(audio_path, model)
    raise ParserError(ErrorCode.ASR_FAILED, f"unsupported ASR_BACKEND: {backend}", stage="asr")


def _resolve_model_name(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def _resolve_coreml_model_repo(model: str) -> str:
    override = os.getenv("ASR_COREML_MODEL_REPO", "").strip()
    if override:
        return override
    resolved = _resolve_model_name(model)
    if "/" in resolved:
        return resolved
    return _COREML_MODEL_REPOS.get(resolved, f"mlx-community/whisper-{resolved}")


def _is_enabled(env_name: str, default: str = "1") -> bool:
    return os.getenv(env_name, default).lower() not in {"0", "false", "no"}


def _default_asr_backend() -> str:
    machine = platform.machine().lower()
    if platform.system() == "Darwin" and machine in {"arm64", "aarch64"}:
        return "coreml"
    return "faster-whisper"


def _build_cookie_header(raw_cookie: str | None, sessdata: str | None) -> str | None:
    cookie = (raw_cookie or "").strip()
    sess = (sessdata or "").strip()

    if not cookie and not sess:
        return None
    if sess.lower().startswith("sessdata="):
        sess = sess.split("=", 1)[1].strip()
    if not sess:
        return cookie

    parts = [p.strip() for p in cookie.split(";") if p.strip()]
    parts = [p for p in parts if not p.lower().startswith("sessdata=")]
    parts.append(f"SESSDATA={sess}")
    return "; ".join(parts)


def _ensure_asr_model_ready(model: str) -> None:
    backend = (os.getenv("ASR_BACKEND", "").strip().lower()) or _default_asr_backend()
    if backend in {"coreml", "mlx"}:
        # mlx-whisper lazily downloads model at first transcription call.
        logger.info("asr warmup skipped for coreml backend (lazy model download)")
        return

    resolved_model = _resolve_model_name(model)
    logger.info("asr warmup start model=%s resolved=%s", model, resolved_model)
    try:
        from faster_whisper.utils import download_model
    except ImportError as exc:
        raise ParserError(
            ErrorCode.ASR_FAILED,
            "faster-whisper not installed; install with `pip install faster-whisper`",
            stage="asr_warmup",
        ) from exc

    hf_home = os.getenv("HF_HOME")
    cache_dir = hf_home if hf_home else None

    try:
        # Fast local-only probe first; this keeps startup instant for warm cache.
        download_model(resolved_model, local_files_only=True, cache_dir=cache_dir)
        logger.info("asr model already cached model=%s", resolved_model)
    except Exception:
        try:
            # Cold start path: allow model download before main parse starts.
            logger.warning("asr model missing; downloading model=%s", resolved_model)
            download_model(resolved_model, local_files_only=False, cache_dir=cache_dir)
            logger.info("asr model downloaded model=%s", resolved_model)
        except Exception as exc:  # noqa: BLE001
            raise ParserError(ErrorCode.ASR_FAILED, f"asr model warmup failed: {exc}", stage="asr_warmup") from exc


async def _parse_url(url: str, sessdata: str | None = None) -> dict[str, Any]:
    timeout_sec = float(os.getenv("HTTP_TIMEOUT_SEC", "12.0"))
    temp_dir = os.getenv("TEMP_DIR", "/tmp")
    max_audio_mb = int(os.getenv("MAX_AUDIO_MB", "200"))
    cookie = _build_cookie_header(os.getenv("BILI_COOKIE"), sessdata or os.getenv("BILI_SESSDATA"))
    asr_model = os.getenv("ASR_MODEL", "whisper-small")

    parser = StandaloneBiliParser(timeout_sec=timeout_sec, temp_dir=temp_dir, max_audio_mb=max_audio_mb, cookie=cookie)
    try:
        return await parser.parse(url, asr_model=asr_model)
    finally:
        await parser.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse text content from a Bilibili video URL")
    parser.add_argument("url", help="Bilibili video URL")
    parser.add_argument("--sessdata", dest="sessdata", default=None, help="Bilibili SESSDATA value")
    args = parser.parse_args()
    logging_ready = False

    try:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_file = os.getenv("LOG_FILE", str(Path(__file__).resolve().parent / "logs" / "min_bili_parse.log"))
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO), handlers=[file_handler], force=True)
        logging_ready = True
        logger.info("logging initialized file=%s level=%s", log_path, log_level)

        if _is_enabled("ASR_WARMUP", "1"):
            _ensure_asr_model_ready(os.getenv("ASR_MODEL", "whisper-small"))
        output = asyncio.run(_parse_url(args.url, sessdata=args.sessdata))
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except httpx.RequestError as exc:
        if logging_ready:
            logger.exception("network request failed")
        # Keep stderr contract machine-readable for callers and shell pipelines.
        err = {
            "code": ErrorCode.UPSTREAM_CONNECT_FAILED,
            "message": f"upstream network error: {exc}",
            "stage": "upstream_request",
        }
        print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
        return 1
    except ParserError as exc:
        if logging_ready:
            logger.exception("parser error code=%s stage=%s", exc.code, exc.stage)
        err = {"code": exc.code, "message": exc.message, "stage": exc.stage}
        print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception:
        if logging_ready:
            logger.exception("unexpected error")
        err = {"code": ErrorCode.INTERNAL_ERROR, "message": "unexpected error"}
        print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
