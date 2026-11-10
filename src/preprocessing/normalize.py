import re
import unicodedata


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
