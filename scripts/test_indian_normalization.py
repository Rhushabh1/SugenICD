
from src.preprocessing.normalize import (
	expand_indian_terms,
	INDIAN_CLINICAL_TERMS,
	MEDICAL_TERMS_HI_MR,
	MEDICAL_TERMS_MR,
)


def test_dictionary(name, dictionary, examples):
	print(f"\n{'=' * 60}")
	print(name)
	print(f"{'=' * 60}")

	for note, expected in examples:
		expanded = expand_indian_terms(
			note,
			dictionaries=[dictionary]
		)

		passed = expected.lower() in expanded.lower()

		print(
			f"{'PASS' if passed else 'FAIL'} | "
			f"{note} -> {expanded}"
		)

		if not passed:
			raise AssertionError(
				f"{name}: expected {expected!r} in {expanded!r}"
			)


# 1. Existing manually curated mappings.
test_dictionary(
	"Manual clinical phrases",
	INDIAN_CLINICAL_TERMS,
	[
		("मधुमेह", "diabetes"),
		("बहुत प्यास", "polydipsia"),
		("seene me jalan", "heartburn"),
	]
)


# 2. Existing Bharatavani glossary.
test_dictionary(
	"Bharatavani glossary",
	MEDICAL_TERMS_HI_MR,
	[
		("उदर", "abdomen"),
	]
)


# 3. Newly extracted Marathi dictionary.
test_dictionary(
	"Extracted Marathi dictionary",
	MEDICAL_TERMS_MR,
	[
		("Madhumeh", "diabetes"),
		("मधुमेह", "diabetes"),
		("Dokedukhi", "headache"),
		("डोकेदुखी", "headache"),
		("Apachan", "indigestion"),
		("अपचन", "indigestion"),
	]
)


print("\nAll independent dictionary tests passed.")
