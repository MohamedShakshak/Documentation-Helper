from doc_helper.config.settings import Settings, RetrievalSettings
from doc_helper.stores.factory import create_vector_store
from doc_helper.retrieval.retriever import Retriever
from doc_helper.evaluation.retrieval_dataset import load_retrieval_dataset
from doc_helper.evaluation.retrieval_metrics import normalize_url

settings = Settings()
store = create_vector_store(settings)
config = RetrievalSettings(search_type="similarity", search_k=16)
retriever = Retriever(store, config)
dataset = load_retrieval_dataset()

for item in dataset:
    q = item["question"]
    relevant = {normalize_url(u) for u in item["relevant_urls"]}
    docs = retriever.retrieve(q)
    retrieved = {normalize_url(doc.metadata.get("source_url", "")) for doc in docs}
    hits = relevant & retrieved
    missed = relevant - retrieved
    if missed:
        print(f"MISS [{item['difficulty']}] {q[:60]}")
        print(f"  wanted:  {sorted(missed)}")
        print(f"  got:     {sorted(retrieved)[:5]}")
        print()