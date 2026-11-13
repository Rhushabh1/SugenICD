
import argparse
import json
import os
from unittest.mock import patch

from src.llm.config import get_provider, get_model
from src.reranking.llm import rerank


class FakeResponse:
	def __init__(self, data):
		self.data = data

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		return False

	def read(self):
		return json.dumps(self.data).encode("utf-8")


def test_auto_router():
	cases = [
		(
			{
				"LLM_PROVIDER": "auto",
				"OPENAI_API_KEY": "mock-openai-key",
				"ANTHROPIC_API_KEY": "mock-anthropic-key"
			},
			"openai"
		),
		(
			{
				"LLM_PROVIDER": "auto",
				"OPENAI_API_KEY": "",
				"ANTHROPIC_API_KEY": "mock-anthropic-key"
			},
			"anthropic"
		),
		(
			{
				"LLM_PROVIDER": "auto",
				"OPENAI_API_KEY": "mock-openai-key",
				"ANTHROPIC_API_KEY": ""
			},
			"openai"
		),
		(
			{
				"LLM_PROVIDER": "auto",
				"OPENAI_API_KEY": "",
				"ANTHROPIC_API_KEY": ""
			},
			"offline"
		)
	]

	for env, expected in cases:
		with patch.dict(os.environ, env, clear=True):
			actual = get_provider()

		if actual != expected:
			raise AssertionError(
				f"Expected {expected}, received {actual}"
			)

		print(f"PASS: auto router -> {expected}")

	# Explicit provider must reject missing credentials.
	with patch.dict(os.environ, {
		"LLM_PROVIDER": "openai",
		"OPENAI_API_KEY": ""
	}, clear=True):
		try:
			get_provider()
			raise AssertionError("Missing OpenAI key was accepted")
		except ValueError:
			print("PASS: missing explicit provider key rejected")


def run_provider_test(provider, candidates):
	test_code = candidates[0]["code"]
	content = json.dumps({"codes": [test_code]})

	if provider == "openai":
		expected_url = "https://api.openai.com/v1/responses"
		expected_key = "mock-openai-key"

		response_data = {
			"output_text": content,
			"usage": {
				"input_tokens": 100,
				"output_tokens": 10
			}
		}

	elif provider == "anthropic":
		expected_url = "https://api.anthropic.com/v1/messages"
		expected_key = "mock-anthropic-key"

		response_data = {
			"content": [
				{"type": "text", "text": content}
			],
			"usage": {
				"input_tokens": 100,
				"output_tokens": 10
			}
		}

	else:
		expected_url = "http://localhost:11434/api/chat"
		expected_key = None

		response_data = {
			"message": {"content": content},
			"load_duration": 0,
			"prompt_eval_count": 100,
			"eval_count": 10
		}

	def fake_urlopen(request, timeout=120):
		if request.full_url != expected_url:
			raise AssertionError(
				f"Wrong endpoint: {request.full_url}"
			)

		if request.get_method() != "POST":
			raise AssertionError("Expected POST request")

		headers = {
			key.lower(): value
			for key, value in request.header_items()
		}

		if provider == "openai":
			if headers.get("authorization") != f"Bearer {expected_key}":
				raise AssertionError("Incorrect OpenAI auth header")

		elif provider == "anthropic":
			if headers.get("x-api-key") != expected_key:
				raise AssertionError("Incorrect Anthropic auth header")

			if headers.get("anthropic-version") != "2023-06-01":
				raise AssertionError("Missing Anthropic version header")

		body = json.loads(request.data.decode("utf-8"))

		if body["model"] != "mock-model":
			raise AssertionError("Incorrect model")

		if provider == "openai":
			schema = body["text"]["format"]
			if schema["type"] != "json_schema":
				raise AssertionError("OpenAI structured output missing")

		elif provider == "anthropic":
			schema = body["output_config"]["format"]
			if schema["type"] != "json_schema":
				raise AssertionError("Anthropic structured output missing")

		elif body["format"]["type"] != "object":
			raise AssertionError("Ollama JSON schema missing")

		return FakeResponse(response_data)

	env = {
		"OPENAI_API_KEY": "mock-openai-key",
		"ANTHROPIC_API_KEY": "mock-anthropic-key",
		"OLLAMA_BASE_URL": "http://localhost:11434"
	}

	with patch.dict(os.environ, env):
		with patch(
			"urllib.request.urlopen",
			side_effect=fake_urlopen
		):
			result = rerank(
				"Patient presents with a test clinical condition.",
				candidates,
				model="mock-model",
				provider=provider
			)

	if result["codes"] != [test_code]:
		raise AssertionError(
			f"Unexpected codes from {provider}: {result['codes']}"
		)

	if result["invalid_codes"]:
		raise AssertionError("Unexpected invalid codes")

	print(f"PASS: {provider} endpoint, headers, schema and parsing")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--provider",
		choices=["ollama", "openai", "anthropic", "all"],
		default="all"
	)
	args = parser.parse_args()

	with open(
		"data/mms-2026-01-enriched.json",
		"r",
		encoding="utf-8"
	) as file:
		catalogue = json.load(file)

	candidates = []

	for entry in catalogue:
		if entry.get("leaf"):
			candidates.append({
				"code": entry["code"],
				"title": entry["title"],
				"score": 1.0
			})

		if len(candidates) == 3:
			break

	if len(candidates) < 3:
		raise ValueError("Could not find three leaf candidates")

	test_auto_router()

	providers = (
		["ollama", "openai", "anthropic"]
		if args.provider == "all"
		else [args.provider]
	)

	for provider in providers:
		run_provider_test(provider, candidates)

	print("\nAll provider tests passed.")


if __name__ == "__main__":
	main()
