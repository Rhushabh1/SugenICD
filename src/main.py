import json
import sys
from pathlib import Path 

from src.preprocessing.normalize import normalize_text


# temporary main file to test docker and i/o formats
# normalizing every note but no predicted codes yet
def main():
	if len(sys.argv) != 3:
		print("Usage: python -m src.main input.jsonl output.jsonl")
		sys.exit(1)

	input_file = Path(sys.argv[1])
	output_file = Path(sys.argv[2])
	output_file.parent.mkdir(parents = True, exist_ok = True)

	count = 0

	with open(input_file, "r", encoding = "utf-8") as infile, open(output_file, "w", encoding = "utf-8") as outfile:
		for line in infile:
			if not line.strip():
				continue

			note = json.loads(line)
			# normalize note
			normalized_text = normalize_text(note["text"])
			# pass norm text to lexical pipeline next
			# temporary output until further predictions are implemented
			prediction = {
						"id": note["id"],
						"codes": []
						}

			outfile.write(json.dumps(prediction) + "\n")
			count+=1

	print(f"Processed notes: {count}")
	print(f"Predictions: {output_file}")


if __name__ == "__main__":
	main()