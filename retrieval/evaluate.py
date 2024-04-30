import pandas as pd
from ir_measures import AP, nDCG, P, Recall, RR, iter_calc
import argparse
from glob import glob
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
args = parser.parse_args()

datapath = "../../../data"

qrels = pd.read_csv(f"{datapath}/qrels/{args.collection}/qrels.txt", sep=" ", header=None,
                    names=["query_id", "doc_id", "relevance"], usecols=[0, 2, 3], dtype={"query_id": str, "doc_id": str})

paths = glob(f"{datapath}/runs/{args.collection}/*.tsv")

measures = [AP]


def compute_measure(run, qrels):
    out = pd.DataFrame(iter_calc(measures, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out


for p in paths:
    run = pd.read_csv(p, names=["query_id", "doc_id", "score"], usecols=[0, 2, 4], sep="\t", dtype={"query_id": str, "doc_id": str})
    out = pd.DataFrame(iter_calc(measures, qrels, run))

    out['measure'] = out['measure'].astype(str)

    filename = p.split("/").pop().rsplit(".", 1)[0]

    out.to_csv(f"{datapath}/measures/{args.collection}/{filename}_{measures[0]}.tsv", sep="\t", index=False)
