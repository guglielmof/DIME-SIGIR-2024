import numpy as np
from .AbstractFilter import AbstractFilter


class CorrelationFilter(AbstractFilter):
    """
    Filter that considers the top-k dimensions of the interaction vector between the query and a relevant document
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.k = kwargs["k"]
        self.simf = "pearson"

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)

        dlist = self.run[(self.run.query_id == query.query_id) & (self.run["rank"] <= self.k)].doc_id.to_list()
        demb = self.docs_encoder.get_encoding(dlist)
        itx_matrix = np.multiply(qemb, demb)

        if self.simf == "pearson":
            stdev = np.array([np.std(itx_matrix, axis=0)])
            stdev = np.matmul(stdev.T, stdev)
            sim = np.multiply(np.cov(itx_matrix.T), 1. / stdev)
        else:
            sim = np.cov(itx_matrix.T)

        sim = sim - np.diag(np.diag(sim))
        return np.median(sim, axis=1)
