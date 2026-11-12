
import argparse
import json
import os
from unittest.mock import patch

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


def run_test(provider, candidates):
	test_code = candidates[0]["code"]
	content = json.dumps({"codes": [test_code]})

	if provider == "openai":
		response_data = {
			"output_text": content,
			"usage": {
				"input_tokens": 100,
				"output_tokens": 10
			}
		}
		expected_url = "https://api.openai.com/v1/responses"

	elif provider == "anthropic":
		response_data = {
			"content": [
				{"type": "text", "text": content}
			],
			"usage": {
				"input_tokens": 100,
				"output_tokens": 10
			}
		}
		expected_url = "https://api.anthropic.com/v1/messages"

	else:
		response_data = {
			"message": {"content": content},
			"load_duration": 0,
			"prompt_eval_count": 100,
			"eval_count": 10
		}
		expected_url = "http://localhost:11434/api/chat"

	def fake_urlopen(request, timeout=120):
		if request.full_url != expected_url:
			raise ValueError("Incorrect API endpoint")

		body = json.loads(request.data.decode("utf-8"))

		if body["model"] != "mock-model":
			raise ValueError("Incorrect model")

		return FakeResponse(response_data)

	note = "Patient presents with a test clinical condition."

	with patch(
		"urllib.request.urlopen",
		side_effect=fake_urlopen
	):
		with patch.dict(os.environ, {
			"OPENAI_API_KEY": "mock-key",
			"ANTHROPIC_API_KEY": "mock-key"
		}):
			result = rerank(
				note,
				candidates,
				model="mock-model",
				provider=provider
			)

	if result["codes"] != [test_code]:
		raise ValueError(
			f"{provider} returned unexpected codes: {result['codes']}"
		)

	if not all(code in [c["code"] for c in candidates]
			   for code in result["codes"]):
		raise ValueError(f"{provider} returned an invalid candidate")

	print(f"PASS: {provider}")
	print("Returned codes:", result["codes"])


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

	providers = (
		["ollama", "openai", "anthropic"]
		if args.provider == "all"
		else [args.provider]
	)

	for provider in providers:
		run_test(provider, candidates)


if __name__ == "__main__":
	main()
