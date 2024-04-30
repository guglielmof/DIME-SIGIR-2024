import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b'}

def search_dense(queries, collection, model, k=1000):

    model = SentenceTransformer(m2hf[model])
    enc_queries = model.encode(queries['query'])

    # Specify the file path of the saved index
    index_filename = f"../../data/indexes/{collection}/faiss/{model}.faiss"
    mapper = list(map(lambda x: x.strip(), open(f"../../data/indexes/{collection}/faiss/{model}.map", "r").readlines()))
    # Load the index from the file
    index = faiss.read_index(index_filename)

    distances, indices = index.search(enc_queries, k)

    out = []
    for i in range(len(distances)):
        normed_distances = distances[i] / np.max(distances[i])
        ranking = np.argsort(normed_distances)
        sorted_docids = [mapper[indices[i][r]] for r in ranking]
        sorted_distances = sorted(normed_distances, key=lambda x: -x)
        out += list(zip([queries.iloc[i]['qid']] * len(distances[i]), sorted_docids, np.arange(len(sorted_distances)), sorted_distances))

    out = pd.DataFrame(out, columns=["qid", "did", "rank", "score"])
    out["Q0"] = "Q0"
    out["run"] = model
    out = out[["qid", "Q0", "did", "rank", "score", "run"]]

    return out
