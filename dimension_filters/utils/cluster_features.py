import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

import scipy.stats as sts

def get_clustering(queries, qrys_encoder, **kwargs):

    docs_encoder = kwargs["docs_encoder"]
    run = kwargs["run"]
    k = kwargs["k"]
    qrys_encoder = qrys_encoder

    return pd.DataFrame({"query_id": queries.query_id, "clustering": queries.apply(lambda x: single_clustering(x, qrys_encoder, run, k, docs_encoder), axis=1)})


def single_clustering(query, qrys_encoder, run, k, docs_encoder, simf = "pearson"):
    qemb = qrys_encoder.get_encoding(query.query_id)

    dlist = run[(run.query_id == query.query_id) & (run["rank"] <= k)].doc_id.to_list()
    demb = docs_encoder.get_encoding(dlist)
    itx_matrix = np.multiply(qemb, demb)
    if simf == "pearson":
        stdev = np.array([np.std(itx_matrix, axis=0)])
        stdev = np.matmul(stdev.T, stdev)
        sim = (1-np.multiply(np.cov(itx_matrix.T), 1./stdev))/2
        '''
        #too slow
        sim = np.zeros((itx_matrix.shape[1], itx_matrix.shape[1]))
        for i in range(itx_matrix.shape[1] - 1):
            for j in range(i+1, itx_matrix.shape[1]):
                corr = sts.pearsonr(itx_matrix[:, i], itx_matrix[:, j])[0]
                sim[i, j], sim[j, i] = corr, corr
        '''
    else:
        sim = np.cov(itx_matrix.T)
    clustering = AgglomerativeClustering(compute_full_tree=True,  compute_distances=True, metric="precomputed", linkage="average").fit(sim)
    features_set = [[{i}, 0] for i in range(768)]

    for c, d in zip(clustering.children_, clustering.distances_):
        features_set.append([features_set[c[0]][0].union(features_set[c[1]][0]), d])

    return features_set