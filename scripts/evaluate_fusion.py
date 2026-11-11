
import json
from pathlib import Path


def load_jsonl(file_path):
	with open(file_path, "r", encoding="utf-8") as file:
		return [json.loads(line) for line in file if line.strip()]


def calculate_metrics(results, gold_notes):
	total = len(results)
	top1 = 0
	top3 = 0

	for result in results:
		gold = set(gold_notes[result["id"]])
		codes = result["codes"][:3]

		if codes and codes[0] in gold:
			top1 += 1

		if any(code in gold for code in codes):
			top3 += 1

	top1_score = top1 / total * 100
	top3_score = top3 / total * 100

	score = (50 * top1 / total) + (25 * top3 / total)

	return {
		"top1": round(top1_score, 2),
		"top3": round(top3_score, 2),
		"score": round(score, 2)
	}


def fuse_candidates(candidates, llm_codes, retrieval_weight, llm_weight):
	llm_ranks = {}

	for rank, code in enumerate(llm_codes, start=1):
		llm_ranks[code] = rank

	scores = []

	for rank, candidate in enumerate(candidates, start=1):
		code = candidate["code"]

		score = retrieval_weight / (60 + rank)

		if code in llm_ranks:
			score += llm_weight / (60 + llm_ranks[code])

		scores.append({
			"code": code,
			"score": score
		})

	scores.sort(key=lambda x: x["score"], reverse=True)

	return [item["code"] for item in scores[:3]]


def main():
	notes = load_jsonl(Path("data/dev_notes.jsonl"))
	details = load_jsonl(Path("out/3_llm_details.jsonl"))

	gold_notes = {note["id"]: note["gold"] for note in notes}

	results = {
		"Hybrid retrieval": [],
		"LLM only": [],
		"Existing fallback": []
	}

	for detail in details:
		note_id = detail["id"]
		candidates = detail["candidates"]
		hybrid_codes = detail["hybrid_codes"]
		llm_codes = detail["llm_codes"]

		results["Hybrid retrieval"].append({
			"id": note_id,
			"codes": [candidate["code"] for candidate in candidates[:3]]
		})

		results["LLM only"].append({
			"id": note_id,
			"codes": llm_codes[:3]
		})

		fallback_codes = list(llm_codes[:3])

		for code in hybrid_codes:
			if len(fallback_codes) == 3:
				break
			if code not in fallback_codes:
				fallback_codes.append(code)

		results["Existing fallback"].append({
			"id": note_id,
			"codes": fallback_codes
		})

	weights = [
		(1.0, 0.0),
		(0.7, 0.3),
		(0.5, 0.5),
		(0.3, 0.7),
		(0.0, 1.0)
	]

	for retrieval_weight, llm_weight in weights:
		name = f"Fusion R{retrieval_weight:.1f}-L{llm_weight:.1f}"
		results[name] = []

		for detail in details:
			codes = fuse_candidates(
				detail["candidates"],
				detail["llm_codes"],
				retrieval_weight,
				llm_weight
			)

			results[name].append({
				"id": detail["id"],
				"codes": codes
			})

	print(f"\nEvaluated notes: {len(details)}")

	print("\nResults:")
	print("-" * 65)
	print(f"{'Method':<25} {'Top-1':>10} {'Top-3':>10} {'Score /75':>12}")
	print("-" * 65)

	for name, predictions in results.items():
		metrics = calculate_metrics(predictions, gold_notes)

		print(
			f"{name:<25} "
			f"{metrics['top1']:>9.2f}% "
			f"{metrics['top3']:>9.2f}% "
			f"{metrics['score']:>11.2f}"
		)

	llm_by_id = {
		result["id"]: result["codes"]
		for result in results["LLM only"]
	}

	fusion_by_id = {
		result["id"]: result["codes"]
		for result in results["Fusion R0.3-L0.7"]
	}

	print("\nNotes where fusion changed the top prediction:")

	for detail in details:
		note_id = detail["id"]
		llm_codes = llm_by_id[note_id]
		fusion_codes = fusion_by_id[note_id]
		gold = gold_notes[note_id]

		if llm_codes[:1] != fusion_codes[:1]:
			print(f"\n{note_id}")
			print("Gold:  ", gold)
			print("LLM:   ", llm_codes)
			print("Fusion:", fusion_codes)


if __name__ == "__main__":
	main()
