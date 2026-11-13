
import argparse
import json
from pathlib import Path


def read_jsonl(path):
	rows = []
	with open(path, "r", encoding="utf-8") as file:
		for line in file:
			if line.strip():
				rows.append(json.loads(line))
	return rows


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--gold", required=True)
	parser.add_argument("--pred", required=True)
	args = parser.parse_args()

	gold_rows = read_jsonl(args.gold)
	pred_rows = read_jsonl(args.pred)

	predictions = {
		row["id"]: row.get("codes", [])
		for row in pred_rows
	}

	groups = {}

	for gold in gold_rows:
		style = gold.get("style", "unknown")
		groups.setdefault(style, []).append(gold)

	groups["ALL"] = gold_rows

	print("\nIndian-language / generalisation evaluation")
	print("============================================")

	for group, rows in groups.items():
		top1 = 0
		top3 = 0
		empty = 0
		missing = 0

		for row in rows:
			codes = predictions.get(row["id"], [])
			gold_codes = set(row["gold"])

			if row["id"] not in predictions:
				missing += 1

			if not codes:
				empty += 1

			if codes and codes[0] in gold_codes:
				top1 += 1

			if gold_codes.intersection(codes[:3]):
				top3 += 1

		total = len(rows)

		print(f"\n{group}: {total} notes")
		print(f"Top-1: {top1}/{total} = {100 * top1 / total:.2f}%")
		print(f"Top-3: {top3}/{total} = {100 * top3 / total:.2f}%")
		print(f"Empty predictions: {empty}")
		print(f"Missing prediction lines: {missing}")


if __name__ == "__main__":
	main()
