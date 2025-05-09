import numpy as np
from .AbstractDime import AbstractDime
import pandas as pd



class Llm(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm_docs = kwargs["llm_docs"]

    def querywise_compute_importance(self, query, *args, **kwargs):
        local_doc = self.llm_docs.query("query_id == @query.query_id").iloc[0]
        importance = np.multiply(query.representation, local_doc.representation)
        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(importance)})
