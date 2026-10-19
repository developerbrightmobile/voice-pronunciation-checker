#!/usr/bin/env python3
"""English pronunciation coach for Hermes Agent.

Pipeline:
  1. get an audio file (Telegram voice note, existing file, or microphone)
  2. transcribe with Groq Whisper (word + segment timestamps)
  3. compare the transcript with an optional expected text
  4. ask an LLM for IPA transcription, per-word verdicts and Russian tips
  5. print / return a structured report

Uses only the Python standard library. ffmpeg is used for audio conversion
and microphone capture.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid

GROQ_BASE = "https://api.groq.com/openai/v1"
HERMES_ENV = "/root/.hermes/.env"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
FALLBACK_LLM_MODEL = "openai/gpt-oss-20b"
LOW_CONFIDENCE_LOGP = -0.6
FFMPEG_CANDIDATES = (
    "ffmpeg",
    "/root/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
)
TOKEN_RE = re.compile(r"[a-z0-9']+")
USER_AGENT = "hermes-voice-pronunciation/1.0 (+https://api.groq.com)"


class PronError(Exception):
    pass


def find_ffmpeg() -> str:
    for candidate in FFMPEG_CANDIDATES:
        if os.path.isabs(candidate):
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    raise PronError(
        "ffmpeg not found. Install ffmpeg or place it in /root/.hermes/tools/."
    )


def load_env_file(path: str | None = None) -> dict:
    path = path or HERMES_ENV
    values = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in values:
                    values[key] = value
    except OSError:
        pass
    return values


def resolve_api_key(cli_key: str | None = None) -> str:
    if cli_key:
        return cli_key
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    values = load_env_file(HERMES_ENV)
    if values.get("GROQ_API_KEY"):
        return values["GROQ_API_KEY"]
    raise PronError(
        "GROQ_API_KEY not found. Pass --api-key, set the env var, "
        f"or add it to {HERMES_ENV}."
    )


def _multipart(fields: list[tuple[str, str]], files: list[tuple]) -> tuple[str, bytes]:
    boundary = "----hermespron" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    for name, filename, content_type, data in files:
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode(
                "utf-8"
            )
        )
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(chunks)


def _request(url: str, data: bytes, headers: dict, timeout: int = 120) -> dict:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PronError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PronError(f"Network error calling {url}: {exc.reason}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PronError(f"Invalid JSON from {url}: {payload[:300]}") from exc


def post_json(url: str, payload: dict, api_key: str, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    return _request(url, body, headers, timeout)


def transcribe(
    audio_path: str,
    api_key: str,
    model: str = DEFAULT_STT_MODEL,
    language: str = "en",
    timeout: int = 180,
) -> dict:
    with open(audio_path, "rb") as handle:
        audio_bytes = handle.read()
    filename = os.path.basename(audio_path)
    fields: list[tuple[str, str]] = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("language", language),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
    ]
    files = [("file", filename, "application/octet-stream", audio_bytes)]
    boundary, body = _multipart(fields, files)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    return _request(f"{GROQ_BASE}/audio/transcriptions", body, headers, timeout)


def normalize_audio(src: str, dst: str | None = None) -> str:
    ffmpeg = find_ffmpeg()
    if dst is None:
        fd, dst = tempfile.mkstemp(suffix=".wav", prefix="hermes_pron_")
        os.close(fd)
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-i",
        src,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PronError(f"ffmpeg failed to convert {src}: {proc.stderr.strip()}")
    return dst


def record_audio(seconds: float, out_path: str, device: str = "default") -> str:
    ffmpeg = find_ffmpeg()
    attempts = [
        ["-f", "pulse", "-i", device],
        ["-f", "alsa", "-i", device],
    ]
    errors = []
    for source in attempts:
        cmd = (
            [ffmpeg, "-v", "error", "-y"]
            + source
            + ["-t", str(seconds), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.getsize(out_path) > 0:
            return out_path
        errors.append(f"{source[1]}: {proc.stderr.strip()[:200]}")
    raise PronError(
        "Microphone capture failed. Tried pulse and alsa. "
        + " | ".join(errors)
        + " — install libasound2/pulseaudio or use --audio with a file."
    )


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def compare_text(hypothesis: str, reference: str) -> dict:
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = 0
    omitted: list[str] = []
    extra: list[str] = []
    substituted: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        elif tag == "delete":
            omitted.extend(ref[i1:i2])
        elif tag == "insert":
            extra.extend(hyp[j1:j2])
        elif tag == "replace":
            ref_block = ref[i1:i2]
            hyp_block = hyp[j1:j2]
            paired = min(len(ref_block), len(hyp_block))
            for offset in range(paired):
                substituted.append(
                    {"expected": ref_block[offset], "heard": hyp_block[offset]}
                )
            if len(ref_block) > paired:
                omitted.extend(ref_block[paired:])
            if len(hyp_block) > paired:
                extra.extend(hyp_block[paired:])
    accuracy = (matched / len(ref) * 100.0) if ref else 0.0
    return {
        "reference_tokens": ref,
        "hypothesis_tokens": hyp,
        "matched": matched,
        "word_count": len(ref),
        "accuracy": round(accuracy, 1),
        "omitted": omitted,
        "extra": extra,
        "substituted": substituted,
    }


def word_confidences(segments: list[dict], words: list[dict]) -> list[dict]:
    result = []
    for word in words or []:
        start = word.get("start", 0.0)
        conf = None
        for seg in segments or []:
            if seg.get("start", 0) <= start < seg.get("end", 0) or (
                start >= seg.get("start", 0) and start <= seg.get("end", 0)
            ):
                conf = seg.get("avg_logprob")
                break
        result.append(
            {
                "word": word.get("word", ""),
                "start": word.get("start"),
                "end": word.get("end"),
                "avg_logprob": conf,
                "low_confidence": conf is not None and conf < LOW_CONFIDENCE_LOGP,
            }
        )
    return result


def average_logprob(segments: list[dict]) -> float | None:
    values = [s.get("avg_logprob") for s in (segments or []) if s.get("avg_logprob") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def confidence_label(avg_logprob: float | None) -> str:
    if avg_logprob is None:
        return "unknown"
    if avg_logprob > -0.35:
        return "high"
    if avg_logprob > LOW_CONFIDENCE_LOGP:
        return "medium"
    return "low"


def build_word_status(comparison: dict, confidences: list[dict]) -> list[dict]:
    status: list[dict] = []
    substituted = {item["expected"] for item in comparison["substituted"]}
    omitted = set(comparison["omitted"])
    low_words = {
        normalize_tokens(c["word"])[0]
        for c in confidences
        if c["low_confidence"] and normalize_tokens(c["word"])
    }
    for token in comparison["reference_tokens"]:
        if token in omitted:
            verdict = "error"
            note = "word appears to be missing"
        elif token in substituted:
            heard = next(
                (s["heard"] for s in comparison["substituted"] if s["expected"] == token),
                "",
            )
            verdict = "error"
            note = f"heard as '{heard}'"
        elif token in low_words:
            verdict = "warn"
            note = "low model confidence"
        else:
            verdict = "ok"
            note = ""
        status.append({"word": token, "status": verdict, "note": note})
    return status


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


SYSTEM_PROMPT = (
    "Ты — опытный преподаватель английского произношения для русскоязычного ученика. "
    "Ты получаешь транскрипцию речи ученика (автоматическое распознавание Whisper), "
    "ожидаемый текст и список слов с пониженной уверенностью распознавания. "
    "Whisper не измеряет произношение напрямую, поэтому опирайся на: различия между "
    "услышанным и ожидаемым, отмеченные слова и типичные ошибки русскоязычных. "
    "Отвечай СТРОГО в формате JSON без пояснений вокруг. Схема:\n"
    "{\n"
    '  "score": <целое 0-100, насколько понятно и правильно произнесено>,\n'
    '  "summary_ru": "<1-2 предложения общей оценки на русском>",\n'
    '  "words": [{"word": "<слово>", "ipa": "<IPA без слэшей>", '
    '"status": "ok|warn|error", "tip": "<краткий совет на русском или пусто>"}],\n'
    '  "tips": ["<общий совет 1>", "<совет 2>"]\n'
    "}\n"
    "Включай в words только значимые слова (без артиклей-дублей), для каждого дай IPA. "
    "Ставь status=error если слово пропущено или распознано как другое, warn — если "
    "произношение похоже, но требует доработки, ok — если всё хорошо. "
    "Советы давай конкретные: губы, язык, долгота гласной, ударение."
)


def llm_analyze(
    transcript: str,
    reference: str,
    comparison: dict | None,
    confidences: list[dict],
    avg_logprob: float | None,
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 120,
) -> dict:
    low_words = [
        c["word"].strip()
        for c in confidences
        if c.get("low_confidence")
    ]
    context = {
        "transcript": transcript,
        "reference": reference or None,
        "accuracy_percent": comparison["accuracy"] if comparison else None,
        "omitted_words": comparison["omitted"] if comparison else None,
        "extra_words": comparison["extra"] if comparison else None,
        "substituted_words": comparison["substituted"] if comparison else None,
        "low_confidence_words": low_words,
        "avg_logprob": avg_logprob,
    }
    user_prompt = (
        "Данные о произношении (JSON):\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\nВерни только JSON по схеме."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }
    try:
        response = post_json(
            f"{GROQ_BASE}/chat/completions", payload, api_key, timeout
        )
    except PronError as exc:
        if model != FALLBACK_LLM_MODEL:
            payload["model"] = FALLBACK_LLM_MODEL
            try:
                response = post_json(
                    f"{GROQ_BASE}/chat/completions", payload, api_key, timeout
                )
            except PronError as exc2:
                raise PronError(f"LLM analysis failed: {exc} / {exc2}") from exc2
        else:
            raise PronError(f"LLM analysis failed: {exc}") from exc
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _extract_json(content)
    if parsed is None:
        raise PronError(f"Could not parse LLM JSON. Raw: {content[:400]}")
    return parsed


def heuristic_analysis(comparison: dict | None, confidences: list[dict]) -> dict:
    words = []
    if comparison:
        low_words = {
            normalize_tokens(c["word"])[0]
            for c in confidences
            if c["low_confidence"] and normalize_tokens(c["word"])
        }
        for item in build_word_status(comparison, confidences):
            if item["word"] in low_words:
                status = item["status"] if item["status"] != "ok" else "warn"
            else:
                status = item["status"]
            words.append(
                {"word": item["word"], "ipa": "", "status": status, "tip": item["note"]}
            )
    score = int(round(comparison["accuracy"])) if comparison else None
    return {
        "score": score,
        "summary_ru": "Автоматический анализ без LLM (только распознавание и сравнение).",
        "words": words,
        "tips": [],
        "engine": "heuristic",
    }


STATUS_ICON = {"ok": "✅", "warn": "⚠️", "error": "❌"}


def render_report(result: dict) -> str:
    analysis = result.get("analysis") or {}
    lines: list[str] = []
    lines.append("🎧 Проверка произношения (English)")
    lines.append("")
    lines.append(f"🎤 Услышано: {result.get('transcript', '').strip()}")
    if result.get("reference"):
        lines.append(f"📖 Ожидалось: {result['reference'].strip()}")
    comparison = result.get("comparison")
    if comparison:
        lines.append(
            f"🎯 Совпадение слов: {comparison['accuracy']}% "
            f"({comparison['matched']}/{comparison['word_count']})"
        )
    conf = result.get("confidence")
    if conf:
        lines.append(
            f"📈 Уверенность модели: {conf['label']} "
            f"(avg_logprob {conf['avg_logprob']})"
        )
    score = analysis.get("score")
    if score is not None:
        lines.append(f"⭐ Оценка: {score}/100")
    lines.append("")

    words = analysis.get("words") or []
    if words:
        lines.append("Слова:")
        for item in words:
            icon = STATUS_ICON.get(item.get("status", "ok"), "•")
            ipa = item.get("ipa") or ""
            ipa_part = f" /{ipa}/" if ipa else ""
            tip = item.get("tip") or ""
            tip_part = f" — {tip}" if tip else ""
            lines.append(f"  {icon} {item.get('word','')}{ipa_part}{tip_part}")
        lines.append("")

    summary = analysis.get("summary_ru")
    if summary:
        lines.append(f"💬 {summary}")
    tips = analysis.get("tips") or []
    if tips:
        lines.append("")
        lines.append("Советы:")
        for tip in tips:
            lines.append(f"  • {tip}")

    omitted = (comparison or {}).get("omitted") or []
    extra = (comparison or {}).get("extra") or []
    if omitted:
        lines.append("")
        lines.append(f"Пропущено: {', '.join(omitted)}")
    if extra:
        lines.append(f"Лишнее: {', '.join(extra)}")
    return "\n".join(lines)


def run(
    audio_path: str,
    reference: str = "",
    language: str = "en",
    api_key: str | None = None,
    stt_model: str = DEFAULT_STT_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    use_llm: bool = True,
    keep_audio: bool = False,
) -> dict:
    key = resolve_api_key(api_key)
    work_path = normalize_audio(audio_path)
    try:
        transcript_data = transcribe(work_path, key, stt_model, language)
    finally:
        if not keep_audio and work_path != audio_path and os.path.exists(work_path):
            os.remove(work_path)

    transcript = transcript_data.get("text", "").strip()
    segments = transcript_data.get("segments") or []
    words = transcript_data.get("words") or []
    confidences = word_confidences(segments, words)
    avg_logprob = average_logprob(segments)
    comparison = compare_text(transcript, reference) if reference.strip() else None

    analysis: dict
    if use_llm:
        try:
            analysis = llm_analyze(
                transcript,
                reference,
                comparison,
                confidences,
                avg_logprob,
                key,
                llm_model,
            )
            analysis.setdefault("engine", llm_model)
        except PronError as exc:
            analysis = heuristic_analysis(comparison, confidences)
            analysis["llm_error"] = str(exc)
    else:
        analysis = heuristic_analysis(comparison, confidences)

    return {
        "transcript": transcript,
        "reference": reference,
        "language": transcript_data.get("language", language),
        "duration": transcript_data.get("duration"),
        "comparison": comparison,
        "confidence": {
            "avg_logprob": avg_logprob,
            "label": confidence_label(avg_logprob),
            "words": confidences,
        },
        "analysis": analysis,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="English pronunciation coach (Groq Whisper + LLM)."
    )
    parser.add_argument("audio", nargs="?", help="path to an audio file")
    parser.add_argument("-r", "--reference", default="", help="expected text")
    parser.add_argument(
        "-R", "--record", type=float, metavar="SECONDS", help="record from microphone"
    )
    parser.add_argument("--device", default="default", help="microphone device")
    parser.add_argument("-l", "--language", default="en", help="spoken language")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-llm", action="store_true", help="skip LLM analysis")
    parser.add_argument("--json", action="store_true", help="print raw JSON only")
    parser.add_argument("--out", default=None, help="write JSON result to a file")
    parser.add_argument("--keep-audio", action="store_true")
    args = parser.parse_args(argv)

    tmp_record = None
    audio = args.audio
    if args.record:
        fd, tmp_record = tempfile.mkstemp(suffix=".wav", prefix="hermes_rec_")
        os.close(fd)
        audio = tmp_record
    if not audio:
        parser.error("provide an audio file or use --record SECONDS")

    try:
        if args.record:
            record_audio(args.record, audio, args.device)
        result = run(
            audio,
            reference=args.reference,
            language=args.language,
            api_key=args.api_key,
            stt_model=args.stt_model,
            llm_model=args.llm_model,
            use_llm=not args.no_llm,
            keep_audio=args.keep_audio,
        )
    except PronError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp_record and os.path.exists(tmp_record) and not args.keep_audio:
            os.remove(tmp_record)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
