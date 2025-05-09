import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import ir_datasets
import argparse
import importlib
import dime.utils
import sys
from multiprocessing.dummy import Pool
sys.path += [".", "DIME_simple/code", "DIME_simple/code/ir_models"]

import local_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", type=str)
    parser.add_argument("-e", "--encoder", type=str)
    parser.add_argument("-d", "--dime", type=str)
    parser.add_argument("--basepath", default=".")
    args = parser.parse_args()


    if args.collection == "trec-dl-2019":
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
        qrels = pd.DataFrame(dataset.qrels_iter())
        queries = pd.DataFrame(dataset.queries_iter()).query("query_id in @qrels.query_id")

    elif args.collection == "trec-dl-2020":
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2020/judged")
        qrels = pd.DataFrame(dataset.qrels_iter())
        queries = pd.DataFrame(dataset.queries_iter()).query("query_id in @qrels.query_id")

    elif args.collection == "trec-robust-2004":
        dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
        qrels = pd.DataFrame(dataset.qrels_iter())
        queries = pd.DataFrame(dataset.queries_iter()).query("query_id in @qrels.query_id")[["query_id", "title"]].rename({"title": "text"}, axis=1)

    else:
        ValueError("collection not recognized")


    col2corpus = {"trec-dl-2019": "msmarco-passages", "trec-dl-2020": "msmarco-passages", "trec-robust-2004": "tipster"}

    encoder = getattr(importlib.import_module(f"ir_models.dense"), args.encoder.capitalize())()
    queries["representation"] = list(encoder.encode_queries(queries.text.to_list()))


    #We assume that you have already computed the memmaps containing the representation for all the documents of the corpus
    #please, visit https://numpy.org/doc/stable/reference/generated/numpy.memmap.html to learn more about memmap.
    #to construct the encoding of the documents, you can use the method encode_documents of the istances of classes in the module ir_models.dense
    #since memmaps do not allow to store the id of the document corresponding to a certain row, we assume this mapping to be stored in a csv file


    memmap_path = f"{args.basepath}/data/memmap/{col2corpus[args.collection]}/{args.encoder}/{args.encoder}.dat"
    memmap_idmp = f"{args.basepath}/data/memmap/{col2corpus[args.collection]}/{args.encoder}/{args.encoder}_map.csv"

    docs_encoder = local_utils.MemmapEncoding(memmap_path, memmap_idmp, embedding_size=768, index_name="doc_id")
    indexWrapper = local_utils.FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())


    if args.dime == "oracle":
        dime_params = {"qrels": qrels, "docs_encoder": docs_encoder}
    if args.dime == "rel":
        dime_params = {"qrels": qrels, "docs_encoder": docs_encoder}
    elif args.dime == "prf":
        run = indexWrapper.retrieve(queries)
        dime_params = {"docs_encoder": docs_encoder, "k": 5, "run": run}
    elif args.dime == "llm":
        # we assume you already have access to a csv (as the one available in the data directory) with llms answers
        answers = pd.read_csv(f"{args.basepath}/data/gpt4_answers.csv").query("query_id in @queries.query_id")
        answers["representation"] = list(encoder.encode_queries(answers.response.to_list()))
        dime_params = {"llm_docs": answers}

    else:
        ValueError("dime not recognized")


    dim_estimator = getattr(importlib.import_module(f"dime"), args.dime.capitalize())(**dime_params)
    importance = dim_estimator.compute_importance(queries)


    def alpha_retrieve(parallel_args):
        importance, queries, alpha = parallel_args
        masked_qembs, r2q = dime.utils.get_masked_encoding(queries, importance, alpha)
        run = indexWrapper.retrieve(masked_qembs, r2q)
        run["alpha"] = alpha
        return run


    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    with Pool(processes=len(alphas)) as pool:
        run = pd.concat(pool.map(alpha_retrieve, [[importance, queries, a] for a in alphas]))

    perf = run.groupby("alpha").apply(
        lambda x: local_utils.compute_measure(x, qrels, ["AP", "R@1000", "MRR", "nDCG@3", "nDCG@10", "nDCG@100", "nDCG@20", "nDCG@50"])) \
        .reset_index().drop("level_1", axis=1)

    print(perf.groupby(["alpha", "measure"]).value.mean().reset_index()\
          .sort_values(["measure", "alpha"], ascending=True)\
          .pivot_table(index="measure", columns="alpha", values="value").to_string())