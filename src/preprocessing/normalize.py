
import re
import unicodedata
import json
import os

from pathlib import Path
from functools import lru_cache


# Paths and glossary loaders

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
	os.getenv("DATA_DIR", ROOT_DIR / "data")
)

GLOSSARY_DIR = DATA_DIR / "medical_glossaries"


def load_medical_terms(filename):
	path = GLOSSARY_DIR / filename

	if not path.exists():
		print(f"Medical glossary not found: {path}")
		return {}

	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, dict):
		print(f"Invalid glossary format: {path}")
		return {}

	print(
		f"Loaded {len(data)} terminology entries from {filename}"
	)

	return data


# Existing Hindi / Marathi terminology glossary.
MEDICAL_TERMS_HI_MR = load_medical_terms(
	"medical_terms_hi_mr.json"
)

# Extracted Marathi dictionary:
# Devanagari Marathi + Romanized Marathi -> English.
MEDICAL_TERMS_MR = load_medical_terms(
	"medical_terms_mr.json"
)


# Manually curated clinical phrases
# Small dictionary for common Indian outpatient clinical phrases.
# These are retrieval hints, not a complete translation system.
INDIAN_CLINICAL_TERMS = {
	# Hindi
	"मधुमेह": "diabetes mellitus",
	"शुगर की बीमारी": "type 2 diabetes mellitus",
	"बहुत प्यास": "increased thirst polydipsia",
	"बार बार पेशाब": "frequent urination polyuria",

	"उच्च रक्तचाप": "hypertension high blood pressure",
	"बीपी बढ़ा": "hypertension high blood pressure",
	"रक्तचाप बढ़ा": "hypertension high blood pressure",
	"पिछली जांच में भी बढ़ा": "repeated elevated blood pressure hypertension",

	"आधे सिर में दर्द": "migraine unilateral headache",
	"सिर के आधे हिस्से में दर्द": "migraine unilateral headache",

	"सीने में जलन": "gastroesophageal reflux heartburn",
	"खट्टी डकार": "acid reflux regurgitation",
	"पानी मुंह में आता है": "regurgitation reflux",

	"खून की कमी": "anaemia anemia",
	"लोह की कमी": "iron deficiency anaemia anemia",

	"घुटनों में दर्द": "knee osteoarthritis joint pain",
	"घुटने में दर्द": "knee osteoarthritis joint pain",
	"जोड़ों में घिसाव": "osteoarthritis joint degeneration",

	# Marathi
	"रक्तदाब वाढलेला": "hypertension high blood pressure",
	"उच्च रक्तदाब": "hypertension high blood pressure",
	"डोक्याच्या एका बाजूला दुखणे": "migraine unilateral headache",
	"अर्धशिशी": "migraine unilateral headache",
	"मागील भेटीतही जास्त": "repeated elevated blood pressure hypertension",

	"छातीत जळजळ": "gastroesophageal reflux heartburn",
	"आंबट ढेकर": "acid reflux regurgitation",
	"गिळताना त्रास नाही": "no dysphagia",

	"रक्ताची कमतरता": "anaemia anemia",
	"लोहाची कमतरता": "iron deficiency anaemia anemia",

	"गुडघेदुखी": "knee osteoarthritis joint pain",
	"गुडघ्यांमध्ये वेदना": "knee osteoarthritis joint pain",
	"सांध्यांची झीज": "osteoarthritis joint degeneration",

	# Common Hinglish / Roman Hindi
	"bahut pyaas": "increased thirst polydipsia",
	"baar baar peshab": "frequent urination polyuria",
	"bp high": "hypertension high blood pressure",
	"aadhe sir me dard": "migraine unilateral headache",
	"seene me jalan": "gastroesophageal reflux heartburn",
	"khun ki kami": "anaemia anemia",
	"ghutne me dard": "knee osteoarthritis joint pain",
}


# normalization
# preprocessing -> removing only whitespaces, lowercase, stripping
def normalize_text(text):
	# normalize unicode chars
	text = unicodedata.normalize("NFKC", text)

	# standardize common punctuations
	text = text.replace("’", "'")
	text = text.replace("‘", "'")
	text = text.replace("–", "-")
	text = text.replace("—", "-")

	# converting to lowercase
	text = text.lower()
	# remove extra whitespaces
	text = re.sub(r"\s+", " ", text)
	return text.strip()



# preserve the original note and append English clinical equivalents
# for supported Hindi, Marathi and Hinglish phrases
# the expanded text is used for retrieval and LLM context
# it does not replace the original note
# Match complete words/phrases instead of accidental substrings.
# This is especially important for short Romanized Marathi keys.
@lru_cache(maxsize=4096)
def get_phrase_pattern(phrase):
	return re.compile(
		r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
	)


def contains_phrase(normalized_note, phrase):
	normalized_phrase = normalize_text(phrase)

	if not normalized_phrase:
		return False

	return bool(
		get_phrase_pattern(normalized_phrase).search(
			normalized_note
		)
	)


# --------------------------------------------------
# Multilingual terminology expansion
# --------------------------------------------------

def expand_indian_terms(note, dictionaries=None):
	"""
	Preserve the original clinical note.

	Append English terminology hints from:
	  1. Manually curated Hindi/Marathi/Hinglish phrases.
	  2. Existing Bharatavani glossary.
	  3. Extracted Marathi-English dictionary.

	These are retrieval hints, not diagnoses or translations
	of the entire clinical note.

	`dictionaries` is optional and is useful for testing
	individual dictionaries independently.
	"""

	normalized_note = normalize_text(note)

	if dictionaries is None:
		dictionaries = [
			INDIAN_CLINICAL_TERMS,
			MEDICAL_TERMS_HI_MR,
			MEDICAL_TERMS_MR,
		]

	matched_terms = []

	for dictionary in dictionaries:
		for phrase, english_terms in sorted(
			dictionary.items(),
			key=lambda item: len(item[0]),
			reverse=True,
		):
			if not contains_phrase(normalized_note, phrase):
				continue

			if isinstance(english_terms, str):
				matched_terms.append(english_terms)

			elif isinstance(english_terms, (list, tuple)):
				matched_terms.extend(
					term
					for term in english_terms
					if isinstance(term, str) and term.strip()
				)

	# Deduplicate exact hints, preserving their original order.
	unique_terms = []
	seen = set()

	for term in matched_terms:
		normalized_term = normalize_text(term)

		if normalized_term and normalized_term not in seen:
			seen.add(normalized_term)
			unique_terms.append(term)

	if not unique_terms:
		return note

	additions = " ".join(unique_terms)

	return (
		f"{note}\n"
		f"Clinical terminology hints: {additions}"
	)
