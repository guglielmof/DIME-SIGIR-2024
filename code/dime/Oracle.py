import numpy as np
from .AbstractDime import AbstractDime
import pandas as pd


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


class Oracle(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.qrels = kwargs["qrels"].copy()

        if kwargs.get("add_non_relevant", False):
            id_pool = self.docs_encoder.get_ids()

            def __add_non_relevant(ds):
                if len(ds) < 3 or len(ds.relevance.unique()) < 2:
                    ndocs = max(len(ds), 2)
                    new_ids = np.random.randint(0, len(id_pool), size = 10*ndocs)
                    new_ids = set([id_pool[i] for i in new_ids])
                    new_ids = list(new_ids.difference(set(ds.doc_id.to_list())))[:ndocs]
                    new_docs = pd.DataFrame({"query_id": [ds.query_id.values[0]] * ndocs, "doc_id": new_ids, "relevance": 0, "iteration": 0})
                    return pd.concat([ds, new_docs])
                else:
                    return ds
            self.qrels = self.qrels.groupby(["query_id"]).apply(__add_non_relevant).reset_index(drop=True)
        self.name = "Oracle"

    def querywise_compute_importance(self, query, *args, **kwargs):
        local_qrels = self.qrels.loc[self.qrels.query_id == query.query_id]
        local_relevance = np.array(local_qrels.relevance.to_list())
        dembs = self.docs_encoder.get_encoding(local_qrels.doc_id.to_list())

        itx_mat = np.multiply(query.representation[np.newaxis, :], dembs)
        corrs = corr2_coeff(itx_mat.T, np.array([local_relevance])).ravel()

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(corrs)})
