import pandas as pd
import argparse
from search_faiss import search_faiss
#from search_pyterrier import search_pyterrier
import string

data_path = "../../../data"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="deeplearning19")
parser.add_argument("-q", "--queries")
parser.add_argument("-m", "--model", default="tasb")
args = parser.parse_args()
# read queries
queries = pd.read_csv(f"{data_path}/queries/{args.collection}/{args.queries}.csv", header=None, names=["qid", "query"], sep=";", dtype={"query":str})

#remove punctuation from queries
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
queries['query'] = queries['query'].apply(lambda x: str(x).translate(translator))

if args.model in ["tasb", "contriever", "ance"]:
    out = search_faiss(queries, args.collection, args.model, index_dir="/ssd/data/faggioli/24-ECIR-FF/data/indexes")
#else:
    #out = search_pyterrier(queries, args.collection, args.model)

out.to_csv(f"{data_path}/runs/{args.collection}/{args.model.replace('_', '-')}_{args.queries}.tsv", header=None, index=None, sep="\t")