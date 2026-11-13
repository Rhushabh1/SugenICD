import json
import argparse
from pathlib import Path

from src.reranking.fusion import fuse_candidates


ROOT = Path(__file__).resolve().parent.parent
# EVALUATION_FILE = ROOT / "out" / "evaluation_enriched_LS5050_RL3070.jsonl"

HIT_K_VALUES = [1, 3, 5, 10, 20, 30]
RETRIEVAL_WEIGHT_CONFIGS = [
	(1.0, 0.0),
	(0.8, 0.2),
	(0.7, 0.3),
	(0.5, 0.5),
	(0.3, 0.7),
	(0.2, 0.8),
	(0.0, 1.0)
]

FUSION_WEIGHT_CONFIGS = [
	(1.0, 0.0),
	(0.8, 0.2),
	(0.7, 0.3),
	(0.5, 0.5),
	(0.3, 0.7),
	(0.2, 0.8),
	(0.0, 1.0)
]

# later experiment with [10, 20, 30, 60, 100]
RETRIEVAL_RRF_K_VALUES = [10, 20, 30, 60, 100]
FUSION_RRF_K_VALUES = [10, 20, 30, 60, 100]

RETRIEVAL_LIMIT = 30


def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate retrieval and LLM fusion configurations."
	)

	parser.add_argument(
		"evaluation_file",
		help="Path to the evaluation JSONL file, relative to project root or absolute."
	)

	return parser.parse_args()


def load_jsonl(file_path):
	with open(file_path, "r", encoding="utf-8") as file:
		return [json.loads(line) for line in file if line.strip()]


def calculate_metrics(rows, prediction_key = None, predictions = None):
	top1 = 0
	top3 = 0

	for i, row in enumerate(rows):
		gold = set(row["gold"])

		if predictions is not None:
			codes = predictions[i][:3]

		else:
			codes = row["predictions"].get(prediction_key, [])[:3]

		if codes and codes[0] in gold:
			top1 += 1

		if any(code in gold for code in codes):
			top3 += 1

	total = len(rows)

	top1_score = top1 / total * 100 if total else 0
	top3_score = top3 / total * 100 if total else 0

	score = (50 * top1 / total) + (25 * top3 / total) if total else 0

	return {
		"top1": round(top1_score, 2),
		"top3": round(top3_score, 2),
		"score": round(score, 2)
	}


def calculate_hit_rates(rows, stage):
	hits = {k: 0 for k in HIT_K_VALUES}

	for row in rows:
		gold = set(row["gold"])

		codes = [candidate["code"] for candidate in row["retrieval"].get(stage, [])]

		for k in HIT_K_VALUES:
			if any(code in gold for code in codes[:k]):
				hits[k] += 1

	total = len(rows)
	return {k: round(hits[k] / total * 100, 2) if total else 0 for k in HIT_K_VALUES}


# combine lexical and semantic rankings using weighted RRF.
# returns the complete ranked union of candidates.
def weighted_rrf(lexical, semantic, lexical_weight=0.5,
				 semantic_weight=0.5, k=60):
	scores = {}
	candidate_details = {}

	# Add lexical contribution
	for rank, candidate in enumerate(lexical, start=1):
		code = candidate["code"]

		scores[code] = scores.get(code, 0.0) + (
			lexical_weight / (k + rank)
		)
		candidate_details[code] = candidate

	# Add semantic contribution
	for rank, candidate in enumerate(semantic, start=1):
		code = candidate["code"]

		scores[code] = scores.get(code, 0.0) + (
			semantic_weight / (k + rank)
		)

		# Preserve the candidate metadata
		if code not in candidate_details:
			candidate_details[code] = candidate

	ranked = []

	for code, score in scores.items():
		candidate = candidate_details[code].copy()
		candidate["weighted_rrf_score"] = score
		ranked.append(candidate)

	ranked.sort(
		key=lambda x: x["weighted_rrf_score"],
		reverse=True
	)

	return ranked



def main():
	args = parse_args()

	evaluation_file = Path(args.evaluation_file)

	if not evaluation_file.is_absolute():
		evaluation_file = ROOT / evaluation_file

	if not evaluation_file.exists():
		print("Evaluation file not found:", evaluation_file)
		print("Run python -m scripts.evaluate_llm first")
		return 

	rows = load_jsonl(evaluation_file)

	if not rows:
		print("No evaluation records found")
		return 

	print("Evaluation file:", evaluation_file)
	print("Loaded notes:", len(rows))
	print("LLM errors:", sum(bool(row["llm"].get("error")) for row in rows))

	# retrieval recall at different candidate depths
	print("\nRetrieval hit rates")
	print("-" * 72)

	print(f"{'Stage':<18}", end = "")

	for k in HIT_K_VALUES:
		print(f"{'Hit@' + str(k):>9}", end="")

	print()
	print("-" * 72)

	for stage, label in [("lexical", "Lexical"),
						("semantic", "Semantic"),
						("hybrid", "Hybrid RRF"),
						("prepared", "After filtering")]:

		hits = calculate_hit_rates(rows, stage)

		print(f"{label:<18}", end = "")

		for k in HIT_K_VALUES:
			print(f"{hits[k]:>8.2f}%", end="")

		print()

	# Experiment: weighted RRF of lexical and semantic retrieval
	print("\nWeighted retrieval RRF experiments")
	print("-" * 90)

	print(
		f"{'Lexical/Semantic':<22} "
		+ "".join(f"{'Hit@' + str(k):>9}" for k in HIT_K_VALUES)
	)

	print("-" * 90)

	retrieval_experiments = []

	for lexical_weight, semantic_weight in RETRIEVAL_WEIGHT_CONFIGS:
		for retrieval_rrf_k in RETRIEVAL_RRF_K_VALUES:

			experiment_results = []

			for row in rows:
				lexical = row["retrieval"].get("lexical", [])
				semantic = row["retrieval"].get("semantic", [])

				ranked = weighted_rrf(
					lexical,
					semantic,
					lexical_weight=lexical_weight,
					semantic_weight=semantic_weight,
					k=retrieval_rrf_k
				)

				experiment_results.append({
					"id": row["id"],
					"codes": [
						candidate["code"]
						for candidate in ranked[:RETRIEVAL_LIMIT]
					]
				})

			# Calculate Hit@K for this experiment
			hits = {k: 0 for k in HIT_K_VALUES}

			for row, result in zip(rows, experiment_results):
				gold = set(row["gold"])
				codes = result["codes"]

				for k in HIT_K_VALUES:
					if any(code in gold for code in codes[:k]):
						hits[k] += 1

			total = len(rows)

			hit_rates = {
				k: round(hits[k] / total * 100, 2)
				if total else 0
				for k in HIT_K_VALUES
			}

			label = (
				f"Retrieval-L{lexical_weight:g}"
				f"-S{semantic_weight:g}"
				f"-K{retrieval_rrf_k}"
			)

			retrieval_experiments.append({
				"label": label,
				"retrieval_lexical_weight": lexical_weight,
				"retrieval_semantic_weight": semantic_weight,
				"retrieval_rrf_k": retrieval_rrf_k,
				"hit_rates": hit_rates,
				"results": experiment_results
			})

			print(
				f"{label:<22}"
				+ "".join(
					f"{hit_rates[k]:>8.2f}%"
					for k in HIT_K_VALUES
				)
			)

	# Best configuration based on Hit@30
	best_retrieval = max(
		retrieval_experiments,
		key=lambda x: (
			x["hit_rates"][30],
			x["hit_rates"][20],
			x["hit_rates"][10]
		)
	)

	print("\nBest retrieval configuration:")
	print("Weights:", best_retrieval["label"])
	print("RRF K:", best_retrieval["retrieval_rrf_k"])
	print("Hit rates:", best_retrieval["hit_rates"])


	# Compare existing methods before trying different fusion weights.
	methods = [
		("Lexical top 3", "lexical"),
		("Semantic top 3", "semantic"),
		("Hybrid top 3", "hybrid"),
		("Prepared top 3", "prepared"),
		("LLM only", "llm"),
		("Existing fallback", "fallback"),
		("Saved fusion", "fused")
	]

	results = []

	for label, key in methods:
		metrics = calculate_metrics(
			rows,
			prediction_key=key
		)

		results.append((label, metrics))

	fusion_experiments = []

	# Replay fusion from saved candidates and LLM rankings.
	for fusion_retrieval_weight, fusion_llm_weight in FUSION_WEIGHT_CONFIGS:
		for fusion_rrf_k in FUSION_RRF_K_VALUES:

			predictions = []

			for row in rows:
				candidates = row["retrieval"].get("prepared", [])
				llm_codes = row["predictions"].get("llm", [])

				codes = fuse_candidates(
					candidates,
					llm_codes,
					retrieval_weight=fusion_retrieval_weight,
					llm_weight=fusion_llm_weight,
					k=fusion_rrf_k
				)

				predictions.append(codes)

			label = (
				f"Fusion-R{fusion_retrieval_weight:g}"
				f"-LLM{fusion_llm_weight:g}"
				f"-K{fusion_rrf_k}"
			)

			metrics = calculate_metrics(
				rows,
				predictions=predictions
			)

			fusion_experiments.append({
				"label": label,
				"fusion_retrieval_weight": fusion_retrieval_weight,
				"fusion_llm_weight": fusion_llm_weight,
				"fusion_rrf_k": fusion_rrf_k,
				"metrics": metrics,
				"predictions": predictions
			})

			results.append((label, metrics))

	print("\nPrediction results")
	print("-" * 68)

	print(
		f"{'Method':<27} "
		f"{'Top-1':>10} "
		f"{'Top-3':>10} "
		f"{'Score /75':>12}"
	)

	print("-" * 68)

	for label, metrics in results:
		print(
			f"{label:<27} "
			f"{metrics['top1']:>9.2f}% "
			f"{metrics['top3']:>9.2f}% "
			f"{metrics['score']:>11.2f}"
		)

	# Inspect notes where fusion changes the LLM top prediction.
	print(
		"\nNotes where saved fusion "
		"changes the LLM top prediction"
	)

	for row in rows:
		llm_codes = row["predictions"].get("llm", [])
		fused_codes = row["predictions"].get("fused", [])

		if llm_codes[:1] != fused_codes[:1]:
			print("\n" + row["id"])
			print("Gold:  ", row["gold"])
			print("LLM:   ", llm_codes)
			print("Fused: ", fused_codes)
			print("Ranks: ", row["gold_ranks"])

	print(
		"\nFusion experiments use the saved LLM output; "
		"no LLM calls were made."
	)

	output_file = evaluation_file.with_name(
	    f"{evaluation_file.stem}_retrieval_and_fusion_experiments.json"
	)

	saved_results = {
	    "evaluation_file": evaluation_file.name,
	    "total_notes": len(rows),
	    "retrieval_experiments": retrieval_experiments,
	    "fusion_experiments": fusion_experiments
	}

	with open(output_file, "w", encoding="utf-8") as file:
	    json.dump(saved_results, file, indent=4)

	print("\nFusion results saved to:", output_file)


if __name__ == "__main__":
	main()