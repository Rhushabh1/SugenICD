# follows the same structure as lexical.py file
import os
import json
import hashlib
import numpy as np

from pathlib import Path
from fastembed import TextEmbedding

from src.preprocessing.normalize import normalize_text

# convert an optional catalogue field into a clean list of strings
def _as_list(value):
	if value is None:
		return []

	if isinstance(value, str):
		values = [value]
	elif isinstance(value, list):
		values = value
	else:
		return []

	result = []
	seen = set()

	for item in values:
		if isinstance(item, str):
			text = item.strip()
		elif isinstance(item, dict):
			# Handle structured entries without assuming one exact schema.
			text = next(
				(
					str(item[key]).strip()
					for key in ("term", "text", "name", "value")
					if item.get(key)
				),
				""
			)
		else:
			continue

		if text and text not in seen:
			result.append(text)
			seen.add(text)

	return result


# format a capped catalogue metadata section
def _section(label, values, limit, max_chars):
	selected = _as_list(values)[:limit]

	if not selected:
		return ""

	content = "; ".join(selected)
	content = content[:max_chars]

	return f"{label}: {content}"


# build a compact semantic representation of one ICD-11 category.
# uses enriched WHO metadata in addition to the category title.
# exclusions are deliberately omitted in this first experiment.
def build_embedding_document(entry, semantic_mode = "enriched"):
	if semantic_mode == "title":
		return normalize_text(entry.get("title", ""))

	if semantic_mode != "enriched":
		raise ValueError(
			f"Unsupported semantic mode: {semantic_mode}"
		)

	sections = []

	title = entry.get("title", "")
	if title:
		sections.append(f"Title: {title}")

	fully_specified_name = entry.get("fully_specified_name", "")
	if fully_specified_name:
		sections.append(
			f"Fully specified name: {str(fully_specified_name)[:500]}"
		)

	index_terms = _section(
		"Index terms",
		entry.get("index_terms"),
		limit=35,
		max_chars=1000
	)
	if index_terms:
		sections.append(index_terms)

	inclusion_terms = _section(
		"Inclusion terms",
		entry.get("inclusion_terms"),
		limit=15,
		max_chars=500
	)
	if inclusion_terms:
		sections.append(inclusion_terms)

	definitions = _section(
		"Definitions",
		entry.get("definitions"),
		limit=2,
		max_chars=500
	)
	if definitions:
		sections.append(definitions)

	coding_notes = _section(
		"Coding notes",
		entry.get("coding_notes"),
		limit=2,
		max_chars=400
	)
	if coding_notes:
		sections.append(coding_notes)

	# Keep each category description reasonably compact.
	document = " | ".join(sections)[:3000]

	return normalize_text(document)


# generate embeddings in batches and save progress so an interrupted
# run can resume from the last completed batch
def generate_embeddings(model, documents, index_file, batch_size=128):
	total = len(documents)

	partial_file = index_file.with_suffix(".partial.npy")
	progress_file = index_file.with_suffix(".progress.json")

	start = 0

	# Resume only when both checkpoint files are valid.
	if partial_file.exists() and progress_file.exists():
		try:
			with open(progress_file, "r", encoding="utf-8") as file:
				progress = json.load(file)

			if (
				progress.get("total") == total
				and progress.get("fingerprint") == index_file.stem
				and np.load(partial_file, mmap_mode="r").shape
					== (total, 384)
			):
				start = progress.get("completed", 0)
				print(f"Resuming from {start}/{total} embeddings...")

		except (OSError, ValueError, json.JSONDecodeError):
			start = 0

	if start == 0:
		embeddings = np.lib.format.open_memmap(
			partial_file,
			mode="w+",
			dtype="float32",
			shape=(total, 384)
		)
	else:
		embeddings = np.lib.format.open_memmap(
			partial_file,
			mode="r+",
			dtype="float32",
			shape=(total, 384)
		)

	for batch_start in range(start, total, batch_size):
		batch_end = min(batch_start + batch_size, total)
		batch = documents[batch_start:batch_end]

		batch_embeddings = np.array(
			list(model.embed(batch, batch_size=64)),
			dtype="float32"
		)

		embeddings[batch_start:batch_end] = batch_embeddings
		embeddings.flush()

		# Save checkpoint only after the batch is safely written.
		with open(progress_file, "w", encoding="utf-8") as file:
			json.dump({
				"total": total,
				"completed": batch_end,
				"fingerprint": index_file.stem
			}, file)

		percent = batch_end * 100 / total
		print(
			f"Embedded {batch_end}/{total} categories "
			f"({percent:.1f}%)",
			flush=True
		)

	# Close the memory-mapped file before renaming it.
	del embeddings

	os.replace(partial_file, index_file)

	if progress_file.exists():
		progress_file.unlink()

	print("Enriched embeddings generated successfully.")

	return np.load(index_file)


# load catalogue and generate embeddings for all category titles
def build_index(mms_file, semantic_mode = "enriched"):
	with open(mms_file, "r", encoding = "utf-8") as file:
		catalogue = json.load(file)

	# 384 dimensional embeddings
	cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
	model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir = cache_dir)

	# Build enriched descriptions instead of embedding titles alone.
	documents = [
		build_embedding_document(entry, semantic_mode)
		for entry in catalogue
	]

	# Fingerprint the actual embedding input. If the catalogue or
	# document representation changes, a different cache is generated.
	fingerprint_source = (semantic_mode + "\n" + "\n".join(documents)).encode("utf-8")
	fingerprint = hashlib.sha256(fingerprint_source).hexdigest()[:12]

	# save catalogue embeddings for reuse (across runs)
	index_dir = Path(os.getenv("INDEX_DIR", Path(mms_file).resolve().parent.parent / "indexes"))
	index_dir.mkdir(parents = True, exist_ok = True)
	
	# index_file = index_dir / f"icd_embeddings_{semantic_mode}_{fingerprint}.npy"
	index_file = index_dir / f"icd_embeddings_{semantic_mode}.npy"

	if index_file.exists():
		# load saved embeddings and verify its dimensions
		embeddings = np.load(index_file)

		if embeddings.shape == (len(catalogue), 384):
			print(f"Loaded enriched semantic cache: {index_file.name}")
		else:
			# if embeddings are missing, recreate
			print("Embedding cache has an invalid shape. Rebuilding embeddings...")
			# titles = [entry["title"] for entry in catalogue]
			embeddings = generate_embeddings(model, documents, index_file)

	else:
		# no index file found, need to create one
		print("Generating catalogue embeddings...")
		# titles = [entry["title"] for entry in catalogue]
		embeddings = generate_embeddings(model, documents, index_file)

		print(f"Saved enriched semantic cache: {index_file.name}")


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
	return model, catalogue, embeddings, parents


# find categories with closest embeddings to the note
def search(note, model, catalogue, embeddings, parents, limit = 20):
	note = normalize_text(note)

	# embed() takes list of queries
	query_embedding = np.array(list(model.embed([note]))[0], dtype = "float32")

	# calculate cosine similarity = AxB/|A||B|
	# matrix mult
	similarities = embeddings @ query_embedding
	# matrix norm
	norms = np.linalg.norm(embeddings, axis = 1)
	query_norm = np.linalg.norm(query_embedding)

	similarities = similarities / np.maximum(norms * query_norm, 1e-12)

	# get indexes of highest scoring categories
	top_indexes = np.argsort(similarities)[::-1][:limit]

	results = []

	for index in top_indexes:
		entry = catalogue[index]

		results.append({
						"code": entry["code"],
						"title": entry["title"],
						"score": float(similarities[index]),
						"parent_code": parents.get(entry["code"])
						})

	return results

