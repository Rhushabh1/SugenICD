import json
from pathlib import Path


def load_jsonl(file_path):
	with open(file_path, "r", encoding="utf-8") as file:
		return [json.loads(line) for line in file if line.strip()]


def main():
	notes = load_jsonl(Path("data/dev_notes.jsonl"))
	details = load_jsonl(Path("out/3_llm_details.jsonl"))

	gold_notes = {
		note["id"]: set(note["gold"])
		for note in notes
	}

	total = 0
	gold_retrieved = 0
	llm_misses = 0
	llm_wrong_top1 = 0

	print("\nLLM ERROR ANALYSIS")
	print("=" * 70)

	for result in details:
		note_id = result["id"]
		gold = gold_notes[note_id]

		candidates = result["candidates"]
		candidate_codes = [candidate["code"] for candidate in candidates]

		llm_codes = result["llm_codes"]
		hybrid_codes = result["hybrid_codes"]
		raw_codes = result["raw_codes"]

		retrieved_gold = gold.intersection(candidate_codes)
		llm_hit = bool(gold.intersection(llm_codes))

		total += 1

		if retrieved_gold:
			gold_retrieved += 1

		if not llm_hit:
			llm_misses += 1

		if llm_codes and llm_codes[0] not in gold:
			llm_wrong_top1 += 1

		if llm_hit:
			continue

		print(f"\nNote: {note_id}")
		print("Gold:", gold)
		print("Gold retrieved:", retrieved_gold)
		print("Hybrid top 3:", hybrid_codes)
		print("LLM selected:", llm_codes)
		print("Raw LLM codes:", raw_codes)

		print("\nGold candidate details:")

		for candidate in candidates:
			if candidate["code"] in gold:
				rank = candidate_codes.index(candidate["code"]) + 1

				print(
					f"Rank {rank}:",
					candidate["code"],
					"|",
					candidate["title"]
				)

		print("\nLLM selected candidate details:")

		for code in llm_codes:
			for candidate in candidates:
				if candidate["code"] == code:
					print(code, "|", candidate["title"])

		print("-" * 70)

	print("\nSUMMARY")
	print("Total notes:", total)
	print("Gold codes retrieved:", gold_retrieved)
	print("LLM misses:", llm_misses)
	print("LLM wrong top-1:", llm_wrong_top1)


if __name__ == "__main__":
	main()
