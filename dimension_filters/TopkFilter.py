import numpy as np
from .AbstractFilter import AbstractFilter


class TopkFilter(AbstractFilter):
    """
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.k = kwargs["k"]

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)
        dlist = self.run[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)].doc_id.to_list()
        demb = np.mean(self.docs_encoder.get_encoding(dlist), axis=0)
        itx_vec = np.multiply(qemb, demb)

        return itx_vec
