import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool


class AbstractFilter:

    def __init__(self, qrys_encoder, **kwargs):
        self.qrys_encoder = qrys_encoder
        self.safe_threading = False

    def filter_dims(self, queries, explode=False, *args, **kwargs):

        if self.safe_threading:
            importance = self._filter_dims_parallel(queries)
        else:
            importance = self.filter_dims(queries)
        if explode:
            importance = importance.explode("importance")
            importance["dim"] = np.tile(np.arange(self.qrys_encoder.shape[1]), len(queries.query_id.unique()))
            importance["drank"] = importance.groupby("query_id")["importance"].rank(method="first", ascending=False).astype(int)

        return importance

    def _filter_dims(self, queries):
        return pd.DataFrame({"query_id": queries.query_id, "importance": queries.apply(self._single_importance, axis=1)})

    def _filter_dims_parallel(self, queries):
        print("started parallel execution")

        with Pool(processes=len(queries.index)) as pool:
            future = [pool.apply_async(self._single_importance, [q]) for _, q in queries.iterrows()]
            out_values = [fr.get() for fr in future]
            #out_values = [self._single_importance(q) for _, q in queries.iterrows()]

        return pd.DataFrame({"query_id": queries.query_id, "importance": out_values})

    def _single_importance(self, query):
        raise NotImplementedError
