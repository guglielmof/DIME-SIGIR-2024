import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys

sys.path.append(".")
sys.path.append("/ssd/data/faggioli/")

import numpy as np
import pandas as pd
from memmap_interface import MemmapCorpusEncoding, MemmapQueriesEncoding
import argparse
import local_utils
import importlib

import retrieval.searching
import retrieval.utils

import dimension_filters
from tqdm import tqdm

from multiprocessing import Pool
import dimension_filters.utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection2corpus = {"deeplearning19": "msmarco-passages", "deeplearning20": "msmarco-passages",
                     "deeplearninghd": "msmarco-passages", "robust04": "tipster"}

tqdm.pandas()

def crossvalidate_results(results, seed=42, reps=30):
    qid_list = results.query_id.unique()
    np.random.seed(seed)
    n_queries = len(qid_list)
    folds = [np.random.choice(np.arange(n_queries), size=n_queries, replace=False) for _ in range(reps)]

    validated_perf = []
    for e, f in enumerate(folds):
        fA = set([qid_list[i] for i in f[:int(n_queries/2)]])
        fB = set([qid_list[i] for i in f[int(n_queries/2):]])
        A = results.loc[results.query_id.isin(fA), ["value", "alpha"]].groupby("alpha").mean().reset_index()
        B = results.loc[results.query_id.isin(fB), ["value", "alpha"]].groupby("alpha").mean().reset_index()

        val_perf = A.merge(B, on=["alpha"], suffixes=["_A", "_B"])
        print(val_perf)
        bestA = val_perf.loc[val_perf["value_A"].idxmax(), "value_B"]
        bestA_alpha= val_perf.loc[val_perf["value_A"].idxmax(), "alpha"]
        bestB = val_perf.loc[val_perf["value_B"].idxmax(), "value_A"]
        bestB_alpha= val_perf.loc[val_perf["value_B"].idxmax(), "alpha"]

        validated_perf.append([e, bestA, bestB, bestA_alpha, bestB_alpha, (bestA+bestB)/2])
    validated_perf = pd.DataFrame(validated_perf, columns=["fold", "best_A", "best_B", "best_A_alpha", "best_B_alpha", "perf"])
    print(validated_perf)




if __name__ == "__main__":
    tqdm.pandas()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="deeplearning20")
    parser.add_argument("-r", "--retrieval_model", default="contriever")
    parser.add_argument("-m", "--measure", default="nDCG@10")
    parser.add_argument("-f", "--filter_function", default="OracularFilter")

    args = parser.parse_args()

    configs = local_utils.get_configs()

    datadir = configs.get("datadir")

    # read the queries
    query_reader_params = {'sep': ";", 'names': ["query_id", "text"], 'header': None, 'dtype': {"query_id": str}}
    queries = pd.read_csv(f"{datadir}/queries/{args.collection}/original.csv", **query_reader_params)

    # read qrels
    qrels_reader_params = {'sep': " ", 'names': ["query_id", "doc_id", "relevance"], "usecols": [0, 2, 3],
                           'header': None, "dtype": {"query_id": str, "doc_id": str}}
    qrels = pd.read_csv(f"{datadir}/qrels/{args.collection}/qrels.txt", **qrels_reader_params)

    # keep only queries with relevant docs
    queries = queries.loc[queries.query_id.isin(qrels.query_id)]

    # load memmap for the corpus
    corpora_memmapsdir = f"/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/memmap/{collection2corpus[args.collection]}/{args.retrieval_model}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/{args.retrieval_model}.dat",
                                        f"{corpora_memmapsdir}/{args.retrieval_model}_map.csv")

    memmapsdir = f"{datadir}/memmaps/{args.retrieval_model}"
    qrys_encoder = MemmapQueriesEncoding(f"{memmapsdir}/{args.collection}/queries.dat",
                                         f"{memmapsdir}/{args.collection}/original_mapping.tsv")


    if args.filter_function == "OracularFilter":
        filtering = dimension_filters.OracularFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder)

    elif args.filter_function == "OriginalOracularFilter":
        # read runs
        run_reader_parmas = {'names': ["query_id", "doc_id"], 'usecols': [0, 2], 'sep': "\t", 'dtype': {"query_id": str, "doc_id": str}}
        run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)

        filtering = dimension_filters.OriginalOracularFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run)

    elif args.filter_function == "RelevantFilter":
        # read runs
        run_reader_parmas = {'names': ["query_id", "doc_id"], 'usecols': [0, 2], 'sep': "\t", 'dtype': {"query_id": str, "doc_id": str}}
        run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)

        filtering = dimension_filters.RelevantFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run)

    elif args.filter_function == "GPTFilter":
        model_classes = {"tasb": "Tasb", "contriever": "Contriever", "ance": "Ance", "tctcolbert": "TctColbert"}
        model = getattr(importlib.import_module("MYRETRIEVE.code.irmodels.dense"), model_classes[args.retrieval_model])()

        filtering = dimension_filters.GPTFilter(qrys_encoder=qrys_encoder, answers_path="../../../data/answers/gpt4_raw.csv", model=model)

    elif args.filter_function == "AbsValueFilter":
        filtering = dimension_filters.AbsValueFilter(qrys_encoder=qrys_encoder)

    elif args.filter_function == "TopkFilter":
        # read runs
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank"], 'usecols': [0, 2, 3], 'sep': "\t",
                             'dtype': {"query_id": str, "doc_id": str, "rank": int}}
        run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)

        filtering = dimension_filters.TopkFilter(docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=1)

    elif args.filter_function == "ScoreStdFilter":
        # read runs
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank"], 'usecols': [0, 2, 3], 'sep': "\t",
                             'dtype': {"query_id": str, "doc_id": str, "rank": int}}
        run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)

        filtering = dimension_filters.ScoreStdFilter(docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=5)

    elif args.filter_function == "CorrelationFilter":
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank"], 'usecols': [0, 2, 3], 'sep': "\t",
                             'dtype': {"query_id": str, "doc_id": str, "rank": int}}
        run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)
        filtering = dimension_filters.CorrelationFilter(docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=1000)

    elif args.filter_function == "LaplacianFilter":
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank"], 'usecols': [0, 2, 3], 'sep': "\t",
                             'dtype': {"query_id": str, "doc_id": str, "rank": int}}
        run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)
        filtering = dimension_filters.LaplacianFilter(docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=100, p=10)

    elif args.filter_function == "WeiBillingsFilter":
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank"], 'usecols': [0, 2, 3], 'sep': "\t",
                             'dtype': {"query_id": str, "doc_id": str, "rank": int}}
        run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)
        filtering = dimension_filters.WeiBillingsFilter(docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=100)

        '''
        filtering = dimension_filters.utils.get_clustering(queries,  qrys_encoder, docs_encoder=docs_encoder, run=run, k=10)

        def get_fset(cluster_struct):
            qemb = qrys_encoder.get_encoding(cluster_struct.query_id)
            tot_mat = []
            clusters = []
            distances = []
            for f in cluster_struct.clustering:
                clus = f[0]
                dist = f[1]
                if len(clus)>20:
                    clusters.append(clus)
                    distances.append(dist)
                    r = np.zeros(768)
                    cons_f = list(clus)
                    r[cons_f] = qemb[cons_f]
                    tot_mat.append(r)

            ip, idx = index.search(np.array(tot_mat), 10)

            nqueries = len(ip)

            local_qrels = qrels.loc[qrels.query_id == cluster_struct.query_id]

            out = []
            for i in range(nqueries):
                local_run = pd.DataFrame({"query_id": cluster_struct.query_id, "doc_id": idx[i], "score": ip[i]})
                local_run.sort_values("score", ascending=False, inplace=True)
                local_run['doc_id'] = local_run['doc_id'].apply(lambda x: mapper[x])
                measure = retrieval.utils.compute_measure(local_run, local_qrels, args.measure)
                out.append([cluster_struct.query_id, clusters[i], distances[i], measure["value"].values[0]])

            out = pd.DataFrame(out, columns=["query_id", "cluster", "distance", "value"])

            return out


        with Pool(processes=len(filtering.index)) as pool:
            future = [pool.apply_async(get_fset, [r]) for _, r in filtering.iterrows()]
            clustering_perf = pd.concat([fr.get() for fr in future])

        ncp = clustering_perf.reset_index().drop("index", axis=1)
        print(ncp.loc[ncp.groupby("query_id")["value"].idxmax()].value.mean())

         #for _, r in filtering.iterrows():
         #   print(r)

        #k = filtering.progress_apply(get_fset, axis=1)
        '''



    else:
        raise ValueError("unrecongized filter")

    '''
        elif args.filter_function == "LetSeeFilter":
            # read runs
            run_reader_parmas = {'names': ["query_id", "doc_id", "rank"], 'usecols': [0, 2, 3], 'sep': "\t",
                                 'dtype': {"query_id": str, "doc_id": str, "rank": int}}
            run = pd.read_csv(f"{datadir}/runs/{args.collection}/{args.retrieval_model}_original.tsv", **run_reader_parmas)

            filtering = dimension_filters.LetSeeFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=5)
        '''

    # load faiss index
    faiss_path = "/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/faiss"
    index, mapper = retrieval.searching.load_index(faiss_path, args.collection, args.retrieval_model)

    rel_dims = filtering.filter_dims(queries, explode=True)

    qembs = qrys_encoder.get_encoding(queries.query_id.to_list())

    q2r = pd.DataFrame({"query_id": queries.query_id.to_list(), "row": np.arange(len(queries.query_id.to_list()))})


    def masked_retrieve_and_evaluate(qembs, dim_importance, alpha):
        n_dims = int(np.round(alpha * qrys_encoder.shape[1]))
        selected_dims = dim_importance.loc[dim_importance["drank"] <= n_dims][["query_id", "dim"]]

        rows = np.array(selected_dims[["query_id"]].merge(q2r)["row"])
        cols = np.array(selected_dims["dim"])

        mask = np.zeros_like(qembs)
        mask[rows, cols] = 1
        enc_queries = np.where(mask, qembs, 0)

        ip, idx = index.search(enc_queries, 1000)
        nqueries = len(ip)

        out = []
        for i in range(nqueries):
            local_run = pd.DataFrame({"query_id": queries.iloc[i]["query_id"], "doc_id": idx[i], "score": ip[i]})
            local_run.sort_values("score", ascending=False, inplace=True)
            local_run['doc_id'] = local_run['doc_id'].apply(lambda x: mapper[x])
            out.append(local_run)

        out = pd.concat(out)

        measure = retrieval.utils.compute_measure(out, qrels, args.measure)

        measure["alpha"] = alpha

        return measure


    alphas = np.round(np.arange(0.2, 1.1, 0.1), 2)

    with Pool(processes=len(alphas)) as pool:
        future = [pool.apply_async(masked_retrieve_and_evaluate, [qembs, rel_dims, alpha]) for alpha in alphas]
        out_values = pd.concat([fr.get() for fr in future])
