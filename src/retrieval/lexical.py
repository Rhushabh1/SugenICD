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
					fully_specified_name,
					index_terms,
					inclusion_terms,
					definitions,
					coding_notes,
					tokenize = "unicode61"
				)
				""")

	# rows = [(entry["code"], entry["title"]) for entry in catalogue]
	rows = []

	for entry in catalogue:
		rows.append((entry["code"], 
					entry["title"], 
					"|".join(entry.get("fully_specified_name", [])),
					"|".join(entry.get("index_terms", [])),
					"|".join(entry.get("inclusion_terms", [])),
					" ".join(entry.get("definitions", [])),
					" ".join(entry.get("coding_notes", [])),
					))

	conn.executemany("""
					INSERT INTO categories (
											code, title, fully_specified_name,
											index_terms, inclusion_terms, 
											definitions, coding_notes 
											) 
					VALUES (?, ?, ?, ?, ?, ?, ?)""",
					rows)
	conn.commit()

	# find parent categories using ICD codes hierarchy
	# need code-to-entry mapping for ICD code splitting to find parent
	code_to_entry = {entry["code"]: entry for entry in catalogue}

	# map of (code -> parent code)
	parents = {}
	for entry in catalogue:
		code = entry["code"]

		# possibly leaf node
		if "." in code:
			parent_code = code.rsplit(".", 1)[0]

			if parent_code in code_to_entry:
				parent = code_to_entry[parent_code]

				# valid depth check
				if parent["depth"] < entry["depth"]:
					parents[code] = parent_code

	# pass details which are used in search()
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
						SELECT code, title, bm25(categories, 0.0, 5.0, 3.0, 3.0, 2.5, 0.5, 1.0), fully_specified_name, index_terms, inclusion_terms
						FROM categories
						WHERE categories MATCH ?
						ORDER BY bm25(categories, 0.0, 5.0, 3.0, 3.0, 2.5, 0.5, 1.0)
						LIMIT ?
						""", (query, limit)).fetchall()

	results = []
	for code, title, score, full_name, index_terms, inclusion_terms in rows:
		searchable_phrases = [title]

		for field in [full_name, index_terms, inclusion_terms]:
			if field:
				searchable_phrases.extend(field.split("|"))

		# check exact matches in category terminology
		exact_match = False

		for phrase in searchable_phrases:
			phrase = normalize_text(phrase.strip())
	
			if len(phrase) > 2 and phrase in text:
				exact_match = True
				break

		# not using parent-child relationship to change predictions
		results.append({
						"code": code,
						"title": title,
						"score": score,
						# will use this metadata later
						"exact_match": exact_match,
						"parent_code": parents.get(code)
						})

	# exact title match are ranked before bm25 match (F, T)
	results.sort(key = lambda x: (not x["exact_match"], x["score"]))

	return results

