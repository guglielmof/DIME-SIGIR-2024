import numpy as np
from .AbstractFilter import AbstractFilter


class RelevantFilter(AbstractFilter):
    """
    Filter that considers the top-k dimensions of the interaction vector between the query and a relevant document
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]

        run = kwargs["run"]
        qrels = kwargs["qrels"]

        self.available_qrels = qrels.merge(run)
        self.available_qrels = self.available_qrels.loc[self.available_qrels.groupby("query_id")["relevance"].idxmax()]
        self.available_qrels = self.pick_relevant(self.available_qrels)

    def pick_relevant(self, available_qrels):
        _pick_first = lambda x: x.iloc[0].doc_id
        picked_docs = available_qrels.groupby("query_id").apply(_pick_first).reset_index().rename({0: "doc_id"}, axis=1)

        return picked_docs.merge(available_qrels)

    def _single_importance(self, query):

        qemb = self.qrys_encoder.get_encoding(query.query_id)

        if query.query_id in self.available_qrels.query_id.values:
            relevant_id = self.available_qrels.loc[self.available_qrels.query_id == query.query_id, "doc_id"].values[0]
            remb = self.docs_encoder.get_encoding(relevant_id)
        else:
            print("relevant not available")
            remb = qemb

        itx_vec = np.multiply(qemb, remb)

        return itx_vec


'''
def relevant_reduce_dim(x, dims=300):
    relevant_id = qrels.loc[(qrels["query_id"] == x.qid)]
    relevant_id = relevant_id.loc[relevant_id.doc_id.isin(run.loc[run.qid == x.qid, "did"])]
    relevant_id = relevant_id.loc[(relevant_id.relevance >= 1)]
    if len(relevant_id.index) == 0:
        remb = qemb
        print("no relevant found")
    else:
        relevant_id = relevant_id.doc_id.values[0]

        remb = docs_encoder.get_encoding(relevant_id)

    itx_vec = np.multiply(qemb, remb)
    best_dims = np.argsort(-itx_vec)[:dims]

    docs = run.loc[(run.qid == x.qid), "did"].to_list()

    demb = docs_encoder.get_encoding(docs)

    scores = np.dot(qemb[best_dims], demb[:, best_dims].T)

    new_rank = pd.DataFrame({"query_id": [x.qid] * len(scores), "doc_id": docs, "score": scores})

    return new_rank
'''
