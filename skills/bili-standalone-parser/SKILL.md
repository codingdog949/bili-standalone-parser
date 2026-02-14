---
name: bili-standalone-parser
description: Parse Bilibili video links into structured JSON text content using a standalone Python script with subtitle-first and ASR fallback behavior. Use when users ask to extract video transcript/content from a Bilibili URL, verify parsing output with SESSDATA/cookie/proxy, troubleshoot upstream connectivity or ASR model warmup, or run reproducible CLI parsing outside project-internal imports.
---

# Bili Standalone Parser

Use the bundled script to parse a Bilibili URL into stable JSON output.

## Quick Start

1. Install dependencies.
```bash
pip install -r scripts/requirements.txt
```

2. Run parser with a URL.
```bash
python scripts/min_bili_parse.py "https://www.bilibili.com/video/BVxxxxxxxxxxx"
```

3. Read outputs.
- `stdout`: parse result JSON (`bvid`, `title`, `source`, `language`, `full_text`, `summary`, `warnings`)
- `stderr`: error JSON on failure
- log file: `./logs/min_bili_parse.log` by default

## Common Operations

### Parse with SESSDATA
```bash
python scripts/min_bili_parse.py "<url>" --sessdata "<SESSDATA_VALUE>"
```

### Parse with local proxy (example 7890)
```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
ALL_PROXY= \
python scripts/min_bili_parse.py "<url>"
```

### Control ASR warmup/model
```bash
ASR_WARMUP=1 ASR_MODEL=whisper-small python scripts/min_bili_parse.py "<url>"
```

## Environment Variables

- `BILI_COOKIE`: full cookie header value
- `BILI_SESSDATA`: SESSDATA value (used if `--sessdata` is not provided)
- `USE_ENV_PROXY`: use proxy variables (`1` default, set `0` to disable)
- `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`: upstream proxy settings
- `ASR_WARMUP`: pre-check/download ASR model before parse (`1` default)
- `ASR_MODEL`: ASR model (`whisper-small` default)
- `HTTP_TIMEOUT_SEC`: upstream timeout seconds (`12.0` default)
- `TEMP_DIR`: temporary directory for audio downloads (`/tmp` default)
- `MAX_AUDIO_MB`: max audio size limit (`200` default)
- `LOG_LEVEL`: log level (`INFO` default)
- `LOG_FILE`: log file path (`./logs/min_bili_parse.log` default)

## Troubleshooting

- If result shows `UPSTREAM_CONNECT_FAILED`, verify DNS/proxy reachability first.
- If parsing is slow and `source=asr`, expect heavy CPU inference time.
- If model warmup fails, check network access to Hugging Face and disk write permission for model cache.
- If subtitle is missing, fallback to ASR is expected behavior.
