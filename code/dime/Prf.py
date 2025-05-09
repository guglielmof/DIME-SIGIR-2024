import numpy as np
from .AbstractDime import AbstractDime
import pandas as pd

class Prf(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.k = kwargs["k"]

    def querywise_compute_importance(self, query, *args, **kwargs):
        local_run = self.run.query("query_id == @query.query_id and rank<@self.k")
        dembs = self.docs_encoder.get_encoding(local_run.doc_id.to_list())
        p_cen = np.mean(dembs, axis=0)

        importance = np.multiply(query.representation, p_cen)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})
