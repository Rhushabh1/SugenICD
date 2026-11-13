
import json
import re
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
	ROOT / "data" / "medical_glossaries" /
	"raw" / "marathi_english_raw.json"
)

OUTPUT_FILE = (
	ROOT / "data" / "medical_glossaries" /
	"medical_terms_mr.json"
)


# Only retain English meanings relevant to clinical notes.
# This deliberately excludes broad/general dictionary meanings.
CLINICAL_GLOSS_RE = re.compile(
	r"\b("
	r"diabetes|"
	r"dysentery|"
	r"tumou?r|"
	r"indigestion|dyspepsia|"
	r"epilepsy|"
	r"paralysis|palsy|"
	r"illness|disease|sickness|"
	r"diarrh(?:ea|oea)|"
	r"constipation|"
	r"deaf|hard of hearing|blind|"
	r"piles|haemorrhoids|hemorrhoids|"
	r"unconsciousness|insensibility|"
	r"nausea|"
	r"fatigue|lassitude|weakness|"
	r"obesity|"
	r"urinate|urination|urine|urinary|"
	r"kidney stone|renal calculus|kidney|renal|"
	r"menstrual discharge|menstruation|menses|"
	r"anaemia|anemia|"
	r"colour-blind|color-blind|"
	r"blood circulation|blood corpuscle|blood group|blood type|"
	r"liver|vagina|vulva|breast|"
	r"itch|rash|"
	r"pain|ache|"
	r"fever|cough|asthma|"
	r"vomit|"
	r"swelling|"
	r"inflammation|infection|"
	r"fracture|wound|ulcer|"
	r"headache|migraine|dizziness|"
	r"fainting|swoon|"
	r"sputum|mucus|"
	r"abdomen|stomach|uterus|"
	r"pregnancy|childbirth|miscarriage|abortion|"
	r"allergy|allergic|"
	r"hypertension|blood pressure|thyroid|palpitation|"
	r"cardiac|heart disease|"
	r"jaundice|malaria|tuberculosis|eczema|"
	r"wheezing|oedema|edema|numbness|tingling|"
	r"tonsillitis|bronchitis|pneumonia|"
	r"chickenpox|measles|cholera|"
	r"dementia|depression|anxiety|insomnia|"
	r"mental disorder|psychosis|hallucination|cancer"
	r")\b",
	re.IGNORECASE
)


# Remove misleading general-language meanings that happen to contain
# words such as blind, kidney, cancer, etc.
EXCLUDED_MEANING_RE = re.compile(
	r"blind man's buff|"
	r"blind imitation|"
	r"blind faith|"
	r"kidney bean|"
	r"sign cancer of zodiac|"
	r"stroke of luck|"
	r"at a stroke|"
	r"to stroke|"
	r"burning coal|"
	r"burning pyre",
	re.IGNORECASE
)


def clean(value):
	if not isinstance(value, str):
		return ""

	return re.sub(r"\s+", " ", value).strip()


def normalize_key(value):
	return clean(value).casefold()


def main():
	if not SOURCE_FILE.exists():
		raise FileNotFoundError(
			f"Source dictionary not found: {SOURCE_FILE}"
		)

	with open(SOURCE_FILE, "r", encoding="utf-8") as f:
		source = json.load(f)

	records = source.get("MarathiEnglishMerged", [])

	print("Source records:", len(records))

	# One lookup dictionary containing both:
	# Romanized Marathi -> English hints
	# Devanagari Marathi -> English hints
	glossary = defaultdict(list)

	selected_records = 0

	for record in records:
		roman_key = clean(record.get("Key"))
		devanagari_key = clean(record.get("Meaning1"))

		english_meanings = []

		for index in range(2, 9):
			meaning = clean(record.get(f"Meaning{index}"))

			if not meaning:
				continue

			if EXCLUDED_MEANING_RE.search(meaning):
				continue

			if CLINICAL_GLOSS_RE.search(meaning):
				english_meanings.append(meaning.lower())

		english_meanings = list(dict.fromkeys(english_meanings))

		if not english_meanings:
			continue

		selected_records += 1

		# Add both Romanized and Devanagari lookup keys.
		for key in (roman_key, devanagari_key):
			normalized_key = normalize_key(key)

			if not normalized_key:
				continue

			for meaning in english_meanings:
				if meaning not in glossary[normalized_key]:
					glossary[normalized_key].append(meaning)

	# Keep the generated dictionary compact.
	final_glossary = {
		key: meanings[:4]
		for key, meanings in glossary.items()
		if meanings
	}

	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

	with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
		json.dump(
			final_glossary,
			f,
			ensure_ascii=False,
			indent=2
		)

	print("Selected source records:", selected_records)
	print("Generated lookup keys:", len(final_glossary))
	print("Output:", OUTPUT_FILE)
	print(
		"Output size:",
		round(OUTPUT_FILE.stat().st_size / 1024, 2),
		"KB"
	)

	print("\nSample mappings:")

	for key in (
		"madhumeh",
		"मधुमेह",
		"dokedukhi",
		"डोकेदुखी",
		"apachan",
		"अपचन",
	):
		if key in final_glossary:
			print(key, "->", final_glossary[key])


if __name__ == "__main__":
	main()
