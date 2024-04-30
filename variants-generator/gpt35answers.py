import argparse
import pandas as pd
import local_utils
from openai import OpenAI
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collections", nargs="+", default=["robust04"])

args = parser.parse_args()

configs = local_utils.get_configs()

datadir = configs.get("datadir")

queries = []
for collection in args.collections:
    qrels = pd.read_csv(f"{datadir}/qrels/{collection}/qrels.txt", sep=" ", header=None,
                        names=["query_id", "doc_id", "relevance"], usecols=[0, 2, 3], dtype={"query_id": str, "doc_id": str})

    query_reader_params = {'sep': ";", 'names': ["qid", "text"], 'header': None, 'dtype': {"qid": str}}
    queries.append(pd.read_csv(f"{datadir}/queries/{collection}/original.csv", **query_reader_params))
    queries[-1] = queries[-1].loc[queries[-1].qid.isin(qrels.query_id)]

queries = pd.concat(queries)
queries = queries.drop_duplicates()

api_key = None
if api_key is None:
    raise ValueError("API key removed. Edit the code and add it.")

client = OpenAI(api_key=api_key)


def ask_response(qry):
    out = None
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": qry["text"]},
            ],
            seed=42
        )
        with open(f"{datadir}/answers/raw/{qry['qid']}.txt", "w") as fp:
            fp.write(str(response))

        out = response.choices[0].message.content
        print(out)

    except Exception as e:
        print(e)
    return out


queries["response"] = queries.apply(ask_response, axis=1)

queries.to_csv(f"{datadir}/answers/gpt4_raw_robust04.csv", index=False)
