import pyterrier as pt

if not pt.started():
    pt.init()




def search_pyterrier(queries, collection, model, k=1000):
    index_path = f"../../data/indexes/{collection}/pyterrier"
    index = pt.IndexFactory.of(f"{index_path}/data.properties")
    retriever = pt.BatchRetrieve(index, wmodel=model)

    out = retriever(queries)
    out["Q0"] = "Q0"
    out["run"] = model.replace('_', '-')
    out = out[["qid", "Q0", "docno", "rank", "score", "run"]]
    return out
