import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from load_index import load_index

# m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b'}
m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "glove": 'sentence-transformers/average_word_embeddings_glove.6B.300d',
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"}


def search_faiss(queries, collection, model_name, k=1000, index_dir="../../data"):
    model = SentenceTransformer(m2hf[model_name])
    enc_queries = model.encode(queries['query'])

    # Specify the file path of the saved index

    index, mapper = load_index(index_dir, collection, model_name)

    innerproducts, indices = index.search(enc_queries, k)
    nqueries = len(innerproducts)
    out = []
    for i in range(nqueries):
        run = pd.DataFrame(list(zip([queries.iloc[i]['qid']] * len(innerproducts[i]), indices[i], innerproducts[i])), columns=["qid", "did", "score"])
        run.sort_values("score", ascending=False, inplace=True)
        run['did'] = run['did'].apply(lambda x: mapper[x])
        run['rank'] = np.arange(len(innerproducts[i]))
        out.append(run)
    out = pd.concat(out)
    out["Q0"] = "Q0"
    out["run"] = model_name.replace('_', '-')
    out = out[["qid", "Q0", "did", "rank", "score", "run"]]

    return out
