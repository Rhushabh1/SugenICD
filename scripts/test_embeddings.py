from fastembed import TextEmbedding


# 384 dimensional embeddings
model = TextEmbedding("BAAI/bge-small-en-v1.5")

texts = ["patient has fever and cough",
		"patient has malaria",
		"patient has vulvovaginal candidosis"]

embeddings = list(model.embed(texts))

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dims: {len(embeddings[0])}")
print(f"First embedding: {embeddings[0][:5]}")