import numpy as np
from .AbstractDime import AbstractDime
import pandas as pd


class Rel(AbstractDime):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.qrels = kwargs["qrels"].copy()
        self.name = "Oracle"

    def querywise_compute_importance(self, query, *args, **kwargs):
        local_qrels = self.qrels.query("query_id in @query.query_id")
        max_rel = local_qrels.relevance.max()
        local_qrels = local_qrels.query("relevance == @max_rel")
        rel_doc = local_qrels.doc_id.iloc[np.random.randint(len(local_qrels))]
        demb = self.docs_encoder.get_encoding(rel_doc)

        relevance = np.multiply(query.representation, demb)

        return pd.DataFrame({"query_id": query.query_id, "dimension": np.arange(self.d), "importance": list(relevance)})
