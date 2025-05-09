import pandas as pd
import numpy as np
import faiss
import ir_measures

class FaissIndex:

    def __init__(self, model_name=None, base_path=None, corpus=None, path=None, data=None, mapper=None):
        self.index_type = "faiss"

        if path is not None:
            self._load_given_path(path)
        elif not (base_path is None or corpus is None or model_name is None):
            path = f"{base_path}/faiss/{corpus}/{model_name}/{model_name}"
            self._load_given_path(path)
        elif not (data is None or mapper is None):
            self._construct_from_data(data, mapper)


        else:
            raise ValueError("when constructing a FaissIndex, either specify the parameter path, or all the parameters base_path, corpus, model_name")

    def _load_given_path(self, path):

        self.index = faiss.read_index(f"{path}.faiss")

        self.mapper = list(map(lambda x: x.strip(), open(f"{path}.map", "r").readlines()))

    def _construct_from_data(self, data, mapper):
        self.index = faiss.IndexFlatIP(data.shape[1])
        self.index.add(data)
        self.mapper = mapper

    def retrieve_and_evaluate(self, queries, qrels=None, measures = None, return_run=True, r2q=None):

        run = self.retrieve(queries, r2q=r2q)

        if qrels is not None and measures is not None:
            measure = evaluate(run, qrels, measures)

        if return_run:
            return run, measure
        else:
            return measure

    def retrieve(self, queries, r2q=None):
        if type(queries) is pd.DataFrame:
            assert "representation" in queries.columns
            qembs = np.array(queries.representation.to_list())
            r2q = {e: q for e, q in enumerate(queries.query_id.to_list())}

        elif type(queries) is np.ndarray:
            if len(queries.shape) == 1:
                qembs = queries.reshape(1, -1)
            else:
                qembs = queries
        else:
            raise TypeError("queries must be a pandas DataFrame or a numpy array")

        ip, idx = self.index.search(qembs, 1000)

        run = pd.DataFrame({"query_id": np.arange(qembs.shape[0]), "doc_id": list(idx), "score": list(ip)}).explode(["doc_id", "score"])

        if r2q is not None:
            run.query_id = run.query_id.map(r2q)

        run.doc_id = run.doc_id.map(lambda x: self.mapper[x])
        run.score = run.score.astype(float)

        run["rank"] = run.groupby("query_id")["score"].rank(ascending=False, method="first").astype(int)

        return run

def evaluate(run, qrels, measures):
    assert measures is not None and type(measures) is list
    measure = compute_measure(run, qrels, measures)
    return measure

def compute_measure(run, qrels, measures, only_available=False):
    """

    :param run: a pd.DataFrame with (at least) three columns: query_id, doc_id, score
    :param qrels: a pd.DataFrame with (at least) three columns: query_id, doc_id, relevancee
    :param measures: a list of either ir_measures measures or strings parsable according to ir_measures
    :param only_available: if True, consider only queries available in the runs

    :return:  a pd.DataFrame with three columns query_id, measure, value.

    This function takes in input a run an the qrels under the form of pd.DataFrame and a list of measures and computes the performance
    query wise. with the only_available paramer, it is possible to enforce that considered queries are only those available in the run.

    """
    if only_available:
        qrels = qrels.loc[qrels.query_id.isin(run.query_id)]

    # parse measures where they are strings
    measures = [ir_measures.parse_measure(m) if type(m) == str else m for m in measures]
    # compute the performance via iter_calc
    performance = pd.DataFrame(ir_measures.iter_calc(measures, qrels, run))
    # cast the measures in the performance dataset into strings
    performance['measure'] = performance['measure'].astype(str)

    return performance
