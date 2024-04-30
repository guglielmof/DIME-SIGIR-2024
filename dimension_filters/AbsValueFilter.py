import numpy as np
from .AbstractFilter import AbstractFilter


class AbsValueFilter(AbstractFilter):
    """
    Filter that considers the top-k dimensions based on their absolute value
    """

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)

        return np.abs(qemb)
