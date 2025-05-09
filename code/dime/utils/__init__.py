import numpy as np
import pandas as pd

def get_masked_encoding(queries: pd.DataFrame, importance: pd.DataFrame, alpha: float) -> np.array:
    """
    This function takes the queries and constructs a masked representation based on the importance with a cutoff alpha
    :param queries:
    :param importance:
    :param alpha:
    :return:
    """
    qembs = np.array(queries.representation.to_list())
    q2r = pd.DataFrame(enumerate(queries.query_id.to_list()), columns=["row", "query_id"])
    r2q = {e: q for e, q in enumerate(queries.query_id.to_list())}

    rep_size = len(queries.representation.values[0])
    n_dims = int(np.round(alpha * rep_size))

    importance["drank"] = importance.groupby("query_id")["importance"].rank(method="first", ascending=False).astype(int)
    selected_dims = importance.loc[importance["drank"] <= n_dims][["query_id", "dimension"]]

    tmp = selected_dims.merge(q2r)
    rows = np.array(tmp["row"])
    cols = np.array(tmp["dimension"])

    mask = np.zeros_like(qembs)
    mask[rows, cols] = 1
    enc_queries = np.where(mask, qembs, 0)

    return enc_queries, r2q
