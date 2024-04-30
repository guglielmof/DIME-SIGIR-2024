import numpy as np
import pandas as pd

from .AbstractFilter import AbstractFilter


class OracularFilter(AbstractFilter):
    """
    Filter that considers the top-k dimensions of the interaction vector between the query and a relevant document
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.qrels = kwargs["qrels"]
        self.docs_encoder = kwargs["docs_encoder"]

    def _single_importance(self, query):
        relevance = np.array(self.qrels.loc[self.qrels.query_id == query.query_id, "relevance"].to_list())

        rel_dids = self.qrels.loc[self.qrels.query_id == query.query_id, "doc_id"].to_list()
        rembs = self.docs_encoder.get_encoding(rel_dids)
        qemb = self.qrys_encoder.get_encoding(query.query_id)

        itx_mat = np.multiply(qemb, rembs)

        corrs = []

        for i in range(itx_mat.shape[1]):
            corrs.append(np.corrcoef(relevance, itx_mat[:, i])[0, 1])

        return corrs
