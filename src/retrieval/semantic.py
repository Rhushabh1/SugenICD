# follows the same structure as lexical.py file
import json
import numpy as np 
from pathlib import Path

from fastembed import TextEmbedding


# load catalogue and generate embeddings for all category titles
def build_index(mms_file):
	with open(mms_file, "r", encoding = "utf-8") as file:
		catalogue = json.load(file)

	# 384 dimensional embeddings
	model = TextEmbedding("BAAI/bge-small-en-v1.5")

	# save catalogue embeddings for reuse (across runs)
	index_dir = Path(mms_file).resolve().parent.parent / "indexes"
	index_dir.mkdir(exist_ok = True)
	index_file = index_dir / "icd_embeddings.npy"

	if index_file.exists():
		# load saved embeddings and verify its dimensions
		embeddings = np.load(index_file)

		if embeddings.shape != (len(catalogue), 384):
			# if embeddings are missing, recreate
			print("Embedding cache is stale. Rebuilding embeddings...")
			titles = [entry["title"] for entry in catalogue]
			embeddings = np.array(list(model.embed(titles)), dtype = "float32")			
			np.save(index_file, embeddings)
			
		else:
			print("Loaded cached catalogue embeddings")

	else:
		# no index file found, need to create one
		print("Generating catalogue embeddings...")
		titles = [entry["title"] for entry in catalogue]
		embeddings = np.array(list(model.embed(titles)), dtype = "float32")

		np.save(index_file, embeddings)


	# find parent categories using URI hierarchy
	# need uri-to-code mapping for uri splitting to find parent
	uri_to_code = {entry["uri"]: entry["code"] for entry in catalogue}

	# map of (code -> parent code)
	parents = {}
	for entry in catalogue:
		parent_uri = entry["uri"].rsplit("/", 1)[0]

		if parent_uri in uri_to_code:
			parents[entry["code"]] = uri_to_code[parent_uri]

	# pass details which are used in search()
	return model, catalogue, embeddings, parents


# find categories with closest embeddings to the note
def search(note, model, catalogue, embeddings, parents, limit = 20):
	# embed() takes list of queries
	query_embedding = np.array(list(model.embed([note]))[0], dtype = "float32")

	# calculate cosine similarity = AxB/|A||B|
	# matrix mult
	similarities = embeddings @ query_embedding
	# matrix norm
	norms = np.linalg.norm(embeddings, axis = 1)
	query_norm = np.linalg.norm(query_embedding)

	similarities = similarities / (norms * query_norm)

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

