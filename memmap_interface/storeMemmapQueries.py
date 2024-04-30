import argparse
import pandas as pd
from glob import glob
from ir_corpora_utils import MsMarco, TIPSTER
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# we need a memmap for the corpus and a memmap for the queries. Additionally, we need a mapping for the corpus and one for each query and its variants

m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "glove": 'sentence-transformers/average_word_embeddings_glove.6B.300d',
        "ance": 'sentence-transformers/msmarco-roberta-base-ance-firstp'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--representation")
    parser.add_argument("-p", "--data_dir", default="../../../data")
    parser.add_argument("-c", "--collection", default="deeplearning19")
    args = parser.parse_args()

    model = SentenceTransformer(m2hf[args.representation])

    queries = []
    qpaths = glob(f"{args.data_dir}/queries/{args.collection}/**/*.csv", recursive=True)
    for qp in qpaths:
        queries.append(pd.read_csv(qp, sep=";", header=None, names=["qid", "text"], dtype={"qid": str}))
        queries[-1]['qtype'] = qp.rsplit("/", 1)[1].rsplit(".", 1)[0]
    queries = pd.concat(queries)
    queries["offset"] = np.arange(len(queries.index))
    # reprs = np.array()
    repr = np.array(queries.text.apply(model.encode).to_list())
    fp = np.memmap(f"{args.data_dir}/memmaps/{args.representation}/{args.collection}/queries.dat", dtype='float32', mode='w+', shape=repr.shape)
    fp[:] = repr[:]
    fp.flush()

    for t in queries.qtype.unique():
        queries.loc[queries.qtype == t, ['qid', 'offset']].to_csv(f"{args.data_dir}/memmaps/{args.representation}/{args.collection}/{t}_mapping.tsv", sep="\t",
                                                                  index=False)
