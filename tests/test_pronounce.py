#!/usr/bin/env python3
"""Unit tests for pronounce.py (no network access required)."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pronounce  # noqa: E402


class TestTokens(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(
            pronounce.normalize_tokens("Hello, World!"), ["hello", "world"]
        )

    def test_apostrophe_kept(self):
        self.assertEqual(pronounce.normalize_tokens("I'm don't"), ["i'm", "don't"])

    def test_numbers_kept(self):
        self.assertEqual(pronounce.normalize_tokens("Grade 5"), ["grade", "5"])

    def test_empty(self):
        self.assertEqual(pronounce.normalize_tokens(""), [])


class TestCompare(unittest.TestCase):
    def test_identical(self):
        cmp = pronounce.compare_text("Hello my name is Kirill", "Hello my name is Kirill")
        self.assertEqual(cmp["accuracy"], 100.0)
        self.assertEqual(cmp["matched"], 5)
        self.assertEqual(cmp["omitted"], [])
        self.assertEqual(cmp["extra"], [])
        self.assertEqual(cmp["substituted"], [])

    def test_omission(self):
        cmp = pronounce.compare_text("hello name is", "hello my name is")
        self.assertIn("my", cmp["omitted"])
        self.assertEqual(cmp["matched"], 3)

    def test_insertion(self):
        cmp = pronounce.compare_text("hello my name is uh kirill", "hello my name is kirill")
        self.assertIn("uh", cmp["extra"])

    def test_substitution(self):
        cmp = pronounce.compare_text("hello my name is cyril", "hello my name is kirill")
        self.assertEqual(
            cmp["substituted"], [{"expected": "kirill", "heard": "cyril"}]
        )

    def test_empty_reference(self):
        cmp = pronounce.compare_text("hello", "")
        self.assertEqual(cmp["accuracy"], 0.0)
        self.assertEqual(cmp["word_count"], 0)
        self.assertIn("hello", cmp["extra"])

    def test_replace_block_unequal_lengths(self):
        cmp = pronounce.compare_text("a b c d", "a b")
        self.assertEqual(cmp["matched"], 2)
        self.assertEqual(cmp["extra"], ["c", "d"])

    def test_replace_block_omitted_words(self):
        cmp = pronounce.compare_text("a b", "a b c d")
        self.assertEqual(cmp["matched"], 2)
        self.assertEqual(cmp["omitted"], ["c", "d"])


class TestConfidence(unittest.TestCase):
    def test_word_confidences_maps_segment(self):
        segments = [{"start": 0, "end": 2, "avg_logprob": -0.9}]
        words = [{"word": "hello", "start": 0.2, "end": 0.8}]
        result = pronounce.word_confidences(segments, words)
        self.assertEqual(result[0]["avg_logprob"], -0.9)
        self.assertTrue(result[0]["low_confidence"])

    def test_word_confidences_high(self):
        segments = [{"start": 0, "end": 2, "avg_logprob": -0.1}]
        words = [{"word": "hi", "start": 0.2, "end": 0.4}]
        self.assertFalse(pronounce.word_confidences(segments, words)[0]["low_confidence"])

    def test_average_logprob(self):
        segments = [{"avg_logprob": -0.2}, {"avg_logprob": -0.4}]
        self.assertEqual(pronounce.average_logprob(segments), -0.3)

    def test_average_logprob_empty(self):
        self.assertIsNone(pronounce.average_logprob([]))

    def test_confidence_labels(self):
        self.assertEqual(pronounce.confidence_label(-0.1), "high")
        self.assertEqual(pronounce.confidence_label(-0.5), "medium")
        self.assertEqual(pronounce.confidence_label(-1.2), "low")
        self.assertEqual(pronounce.confidence_label(None), "unknown")


class TestWordStatus(unittest.TestCase):
    def test_statuses(self):
        cmp = pronounce.compare_text(
            "hello my name is cyril", "hello my name is kirill"
        )
        conf = [{"word": "my", "low_confidence": True}, {"word": "name", "low_confidence": False}]
        status = {item["word"]: item["status"] for item in pronounce.build_word_status(cmp, conf)}
        self.assertEqual(status["hello"], "ok")
        self.assertEqual(status["my"], "warn")
        self.assertEqual(status["kirill"], "error")


class TestJsonExtraction(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(pronounce._extract_json('{"a": 1}'), {"a": 1})

    def test_code_fence(self):
        self.assertEqual(pronounce._extract_json('```json\n{"a": 2}\n```'), {"a": 2})

    def test_embedded(self):
        self.assertEqual(pronounce._extract_json('text {"a": 3} tail'), {"a": 3})

    def test_invalid(self):
        self.assertIsNone(pronounce._extract_json("no json here"))


class TestEnv(unittest.TestCase):
    def test_load_env_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as handle:
            handle.write("# comment\nGROQ_API_KEY=abc123\nQUOTED=\"xyz\"\n")
            path = handle.name
        try:
            values = pronounce.load_env_file(path)
            self.assertEqual(values["GROQ_API_KEY"], "abc123")
            self.assertEqual(values["QUOTED"], "xyz")
        finally:
            os.remove(path)

    def test_resolve_api_key_precedence_cli(self):
        self.assertEqual(pronounce.resolve_api_key("cli-key"), "cli-key")

    def test_resolve_api_key_env(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "env-key"}, clear=False):
            self.assertEqual(pronounce.resolve_api_key(None), "env-key")

    def test_resolve_api_key_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as handle:
            handle.write("GROQ_API_KEY=file-key\n")
            path = handle.name
        try:
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                pronounce, "HERMES_ENV", path
            ):
                self.assertEqual(pronounce.resolve_api_key(None), "file-key")
        finally:
            os.remove(path)

    def test_resolve_api_key_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            pronounce, "HERMES_ENV", "/nonexistent/.env"
        ):
            with self.assertRaises(pronounce.PronError):
                pronounce.resolve_api_key(None)


class TestMultipart(unittest.TestCase):
    def test_structure(self):
        fields = [("model", "whisper-large-v3-turbo")]
        files = [("file", "a.wav", "application/octet-stream", b"BYTES")]
        boundary, body = pronounce._multipart(fields, files)
        self.assertIn(boundary.encode(), body)
        self.assertIn(b'name="model"', body)
        self.assertIn(b"whisper-large-v3-turbo", body)
        self.assertIn(b'filename="a.wav"', body)
        self.assertIn(b"BYTES", body)
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))


class TestRender(unittest.TestCase):
    def test_render_contains_sections(self):
        result = {
            "transcript": "hello my name is kirill",
            "reference": "hello my name is kirill",
            "comparison": {
                "accuracy": 100.0,
                "matched": 5,
                "word_count": 5,
                "omitted": [],
                "extra": [],
                "substituted": [],
            },
            "confidence": {"label": "high", "avg_logprob": -0.2, "words": []},
            "analysis": {
                "score": 95,
                "summary_ru": "Хорошо",
                "words": [{"word": "kirill", "ipa": "kɪˈrɪl", "status": "ok", "tip": ""}],
                "tips": ["tip one"],
            },
        }
        text = pronounce.render_report(result)
        self.assertIn("Проверка произношения", text)
        self.assertIn("95/100", text)
        self.assertIn("/kɪˈrɪl/", text)
        self.assertIn("tip one", text)


class TestHeuristic(unittest.TestCase):
    def test_heuristic_status(self):
        cmp = pronounce.compare_text("hello world", "hello word")
        result = pronounce.heuristic_analysis(cmp, [])
        self.assertEqual(result["engine"], "heuristic")
        self.assertEqual(result["score"], 50)

    def test_heuristic_no_comparison(self):
        result = pronounce.heuristic_analysis(None, [])
        self.assertIsNone(result["score"])


class TestFfmpeg(unittest.TestCase):
    def test_find_ffmpeg(self):
        path = pronounce.find_ffmpeg()
        self.assertTrue(os.path.exists(path))


@unittest.skipUnless(
    os.environ.get("HERMES_PRON_IT") == "1",
    "set HERMES_PRON_IT=1 to enable the live Groq API test",
)
class TestIntegration(unittest.TestCase):
    def test_transcribe_real_audio(self):
        audio = os.environ.get("HERMES_PRON_AUDIO", "/tmp/sample.wav")
        if not os.path.exists(audio):
            self.skipTest(f"audio fixture not found: {audio}")
        key = pronounce.resolve_api_key(None)
        work = pronounce.normalize_audio(audio)
        try:
            data = pronounce.transcribe(work, key)
        finally:
            if os.path.exists(work) and work != audio:
                os.remove(work)
        self.assertTrue(data.get("text", "").strip())
        self.assertTrue(data.get("segments"))

    def test_full_pipeline_real_audio(self):
        audio = os.environ.get("HERMES_PRON_AUDIO", "/tmp/sample.wav")
        if not os.path.exists(audio):
            self.skipTest(f"audio fixture not found: {audio}")
        result = pronounce.run(
            audio,
            reference="Hello, my name is Kirill and I am learning English pronunciation",
            use_llm=True,
        )
        self.assertGreater(result["comparison"]["accuracy"], 50.0)
        self.assertIsNotNone(result["analysis"].get("score"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
