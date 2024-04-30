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

    corpus = "msmarco-passage" if args.collection in ["deeplearning19", "deeplearning20"] else "tipster"

    # import the runs and join them to have a single list of documents
    fpaths = glob(f"{args.data_dir}/runs/{args.collection}/**/*.tsv", recursive=True)
    fpaths = [f for f in fpaths if f.rsplit("/")[-1].split("_", 1)[0] == args.representation]

    runs = []
    for f in fpaths:
        runs.append(pd.read_csv(f, sep="\t", names=["qid", "_", "did", "rank", "score", "model"], header=None, dtype={"did": str}))
    runs = pd.concat(runs)

    docs_list = runs.did.unique()

    if corpus == "msmarco-passage":
        coll = MsMarco(path="/home/ims/mnt_grace/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/MSMARCO-PASSAGES/collection/collection.tsv")
    else:
        coll = TIPSTER(path="/home/ims/mnt_grace/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/IR_DATASETS_CORPORA")

    docs = pd.DataFrame([[d, coll.get_doc(d)] for d in docs_list], columns=["did", "text"])
    docs = docs.sort_values("did")
    docs["offset"] = np.arange(len(docs.index))
    model = SentenceTransformer(m2hf[args.representation])

    # Start the multiprocess pool on all available CUDA devices

    fp = np.memmap(f"{args.data_dir}/memmaps/{args.representation}/corpora/{corpus}/corpus.dat", dtype='float32', mode='w+',
                   shape=(len(docs_list), 768))

    step = 6000

    pool = model.start_multi_process_pool()
    for i in tqdm(range(0, len(docs.text.to_list()), step)):
        # Compute the embeddings using the multiprocess pool
        fp[i:i + step] = model.encode_multi_process(docs.text.to_list()[i:i + step], pool)

    model.stop_multi_process_pool(pool)

    docs[["did", "offset"]].to_csv(f"{args.data_dir}/memmaps/{args.representation}/corpora/{corpus}/corpus_mapping.csv", index=False)