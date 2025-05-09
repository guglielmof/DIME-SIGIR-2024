import numpy.random as npr
import numpy as np
import pandas as pd

def crossvalidation(perf_table, k=5, seed=42):
    local_perf_table = perf_table.copy()
    npr.seed(seed)
    query_ids = np.array(local_perf_table.query_id.unique())

    #this is to shuffle the query ids
    query_ids = query_ids[npr.choice(np.arange(len(query_ids)), len(query_ids), replace=False)]
    fold_ids = np.array_split(query_ids, k)
    fold_ids = {query_id: e for e, fold in enumerate(fold_ids) for query_id in fold}
    local_perf_table["fold"] = local_perf_table["query_id"].map(fold_ids)
    fold_performance = local_perf_table.groupby(["alpha", "fold"])["value"].mean().reset_index()

    fold_cv_perf = []
    for fold in fold_performance.fold.unique():
        best_aplha = fold_performance.loc[fold_performance.fold!=fold].groupby("alpha")["value"].mean().idxmax()
        best_performance = local_perf_table.loc[(local_perf_table.fold==fold) & (local_perf_table.alpha==best_aplha), ["query_id", "value"]]
        fold_cv_perf.append(best_performance)
    fold_cv_perf = pd.concat(fold_cv_perf)
    return fold_cv_perf