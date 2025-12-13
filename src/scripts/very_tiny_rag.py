import filter_stderror # noqa: F401 (redefined sys.stderr)
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F

model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
model_path_saved = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2/saved"

model = SentenceTransformer(model_path, local_files_only=True)

# 1️⃣ Preprocessing (done once)
corpus = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Moscow is the capital of Russia.",
    "Tokyo is the capital of Japan",
    "Tokyo has a population of over 13 million.",
    # ... thousands/millions of chunks
]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 2️⃣ At query time (per user request)
user_query = ["What is the capital of France?"]
query_vector = model.encode(user_query, convert_to_tensor=True)

# 3️⃣ Retrieve
scores = util.cos_sim(query_vector, corpus_embeddings)[0]
print(f"scores: {scores}")
top_k = torch.topk(scores, k=3)  # e.g., k=3
retrieved_docs = [corpus[i] for i in top_k.indices]

# 3a
q_norm = F.normalize(query_vector, p=2, dim=1)      # (2, 768)
c_norm = F.normalize(corpus_embeddings, p=2, dim=1)      # (5, 768)
S2 = torch.mm(q_norm, c_norm.T)          # (2, 5) — identical to util.cos_sim

scores1 = S2[0]
print(f"scores1: {scores1}")

# retrieved_docs → ["Paris is the capital of France.", ...]

# 4️⃣ Augment & Generate
context = "\n".join(retrieved_docs)
prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {user_query}

Answer:"""

print(f"prompt: {prompt}")
#response = llm.generate(prompt)  # e.g., via transformers, vLLM, or API