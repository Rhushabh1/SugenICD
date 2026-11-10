import json
import re
import sqlite3

from src.preprocessing.normalize import normalize_text


STOP_WORDS = {"the", "and", "or", "a", "an", "is", "was", "were", "of", "to", "in", "on", "at", "for", "with", "by", "from", "patient", "reports", "reported"}

# adding clinical synonyms
CLINICAL_SYNONYMS = {
					"thrush": ["candidosis"],
					"urti": ["upper", "respiratory", "infection"],
					"jiggers": ["tungiasis"],
					"chickenpox": ["varicella"],
					"tinea": ["dermatophytosis"],
					"oa": ["osteoarthritis"],
					"copd": ["pulmonary", "obstructive", "disease"],
					"htn": ["hypertension"],
					"dm2": ["diabetes", "mellitus"],
					"vaginal": ["vulvovaginal"],
					"vulval": ["vulvovaginal"],
					}


# loading ICD catalogue directly from mms file
def build_index(mms_file):
	with open(mms_file, "r", encoding = "utf-8") as file:
		catalogue = json.load(file)

	# creating an in-memory sqlite fts5 index of all category titles
	conn = sqlite3.connect(":memory:")
	conn.execute("""
				CREATE VIRTUAL TABLE categories USING fts5(
					code UNINDEXED,
					title,
					tokenize = "unicode61"
				)
				""")

	rows = [(entry["code"], entry["title"]) for entry in catalogue]
	conn.executemany("INSERT INTO categories (code, title) VALUES (?, ?)",
					rows)
	conn.commit()

	# find parent categories using URI hierarchy
	# need uri-to-code mapping for uri splitting to find parent
	uri_to_code = {entry["uri"]: entry["code"] for entry in catalogue}

	# map of (code -> parent code)
	parents = {}
	for entry in catalogue:
		parent_uri = entry["uri"].rsplit("/", 1)[0]

		if parent_uri in uri_to_code:
			parents[entry["code"]] = uri_to_code[parent_uri]

	return conn, parents


# extract searchable terms fron note & use bm25 to rank titles
def search(note, conn, parents, limit = 20):
	text = normalize_text(note)

	# extract unique searchable terms -> to add STOP words later
	# tokenize note
	tokens = re.findall(r"[a-z0-9]+", text)
	# remove the stop words from tokens
	tokens = list(dict.fromkeys(token for token in tokens if token not in STOP_WORDS))

	if not tokens: 
		return []

	# no truncation of tokens, diagnosis may appear near the end of note text
	# # keep the search query small for now -> to expand later
	# tokens = tokens[:50]

	# expand common clinical terms
	expanded_tokens = []
	for token in tokens:
		expanded_tokens.append(token)

		if token in CLINICAL_SYNONYMS:
			expanded_tokens.extend(CLINICAL_SYNONYMS[token])

	tokens = list(dict.fromkeys(expanded_tokens))

	# construct query for bm25 matching
	query = " OR ".join(f'"{token}"' for token in tokens)

	# bm25 score -> lower is better
	rows = conn.execute("""
						SELECT code, title, bm25(categories)
						FROM categories
						WHERE categories MATCH ?
						ORDER BY bm25(categories)
						LIMIT ?
						""", (query, limit)).fetchall()

	results = []
	for code, title, score in rows:
		norm_title = normalize_text(title)
		
		# not using parent-child relationship to change predictions
		results.append({
						"code": code,
						"title": title,
						"score": score,
						# will use this metadata later
						"exact_match": norm_title in text,
						"parent_code": parents.get(code)
						})

	# exact title match are ranked before bm25 match (F, T)
	results.sort(key = lambda x: (not x["exact_match"], x["score"]))

	return results

