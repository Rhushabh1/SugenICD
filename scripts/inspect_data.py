import json
from pathlib import Path 
from collections import Counter


# basic smoke testing all the provided files for inspection
ROOT = Path(__file__).resolve().parent.parent

NOTES_FILE = ROOT / "data" / "dev_notes.jsonl"
MMS_FILE = ROOT / "data" / "mms-2026-01.json"
SCORER_FILE = ROOT / "score.py"


# loading the dev notes
notes = []
with open(NOTES_FILE, "r", encoding = "utf-8") as file:
	for line in file:
		if line.strip():
			notes.append(json.loads(line))

# loading the icd catalogue
catalogue = []
with open(MMS_FILE, "r", encoding = "utf-8") as file:
	catalogue = json.load(file)

# basic note stats
note_ids = [note["id"] for note in notes]
gold_codes = [code for note in notes for code in note["gold"]]
styles = Counter(note["style"] for note in notes)

# basic catalogue stats
catalogue_codes = [entry["code"] for entry in catalogue]
chapters = Counter(entry["chapter"] for entry in catalogue)
leaf_count = sum(entry["leaf"] for entry in catalogue)
residual_count = sum(entry["residual"] for entry in catalogue)

# printing stats
print("\nData Inspection ->")

print(f"\n{NOTES_FILE}")
print(f"Total notes: {len(notes)}")
print(f"Unique note IDs: {len(set(note_ids))}")
print(f"Total gold labels: {len(gold_codes)}")
print(f"Unique gold codes: {len(set(gold_codes))}")
print(f"Note styles ->")
for style, count in styles.items():
	print(f"{style}: {count}")

print(f"\n{MMS_FILE}")
print(f"Total categories: {len(catalogue)}")
print(f"Unique codes: {len(set(catalogue_codes))}")
print(f"Leaf categories: {leaf_count}")
print(f"Non-leaf categories: {len(catalogue) - leaf_count}")
print(f"Residual categories: {residual_count}")
print(f"Chapters: {len(chapters)}")
print(f"Categories per chapter ->")
for chapter, count in sorted(chapters.items()):
	print(f"chapter {chapter}: {count}")

# check whether every gold code exists in the catalogue?
missing_codes = set(gold_codes) - set(catalogue_codes)
print(f"\nMissing codes: {missing_codes}")

print("\nSample notes ->")
for note in notes[:5]:
	print(f"{note["id"]} | {note["style"]} | text len: {len(note["text"])} | gold: {note["gold"]}")

print(f"\nScore.py: {SCORER_FILE.exists()}\n")




