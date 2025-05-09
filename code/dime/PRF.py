import numpy as np
from .AbstractDime import AbstractDime
import pandas as pd

class PRF(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.k = kwargs["k"]

    def querywise_compute_importance(self, query, *args, **kwargs):
        local_run = self.run.query("query_id == @query.query_id and rank<@self.k")
        print(local_run)
        dembs = self.docs_encoder.get_encoding(local_run.doc_id.to_list())

        itx_mat = np.multiply(query.representation[np.newaxis, :], dembs)

        importance = np.mean(itx_mat, axis=0)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})
