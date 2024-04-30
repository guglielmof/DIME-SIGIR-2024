from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.parsing.porter import PorterStemmer
from gensim.utils import deaccent
import gensim
import string
from ir_corpora_utils import MsMarco
from collections import Counter
from tqdm import tqdm
import math
from multiprocessing import Pool
import itertools
import numpy as np

coll = MsMarco(path="/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/MSMARCO-PASSAGES/collection/collection.tsv")

stemmer = PorterStemmer()


def process_document(d):
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    translator = str.maketrans(punctuation, ' ' * len(punctuation))

    s = deaccent(d.translate(translator).replace("-", " ").encode('ascii', 'ignore'))

    return dict(Counter(stemmer.stem_sentence(gensim.parsing.preprocessing.remove_stopwords(s)).split()))


coll.corpus = coll.corpus.dropna()
docs = list(zip(coll.corpus.index, coll.corpus["text"]))


def chunk_based_on_number(lst, chunk_numbers):
    n = math.ceil(len(lst) / chunk_numbers)

    chunks = []
    for x in range(0, len(lst), n):
        each_chunk = lst[x: n + x]
        chunks.append(each_chunk)

    return chunks


workers = 72
docs_splitted = chunk_based_on_number(docs, workers)


def _process_docs(docs_list):
    processed_docs = []
    vocab = []
    for k, v in tqdm(docs_list):
        pd = process_document(v)
        processed_docs.append((k, pd))
        vocab += pd.keys()
    return [vocab, processed_docs]


with Pool(processes=workers) as pool:
    future = [pool.apply_async(_process_docs, [c]) for c in docs_splitted]
    docs = [fr.get() for fr in future]

vocab = set(list(itertools.chain.from_iterable([d[0] for d in docs])))
docs = list(itertools.chain.from_iterable([d[1] for d in docs]))
ids = [d[0] for d in docs]
docs = [d[1] for d in docs]
d2id = {d: e for e, d in enumerate(ids)}

tot_freqs = {v: 0 for v in vocab}
for d in docs:
    for w in d:
        tot_freqs[w] += d[w]

rare_words = set([w for w in tot_freqs if tot_freqs[w] <= 5])

vocab = vocab.difference(rare_words)

v2id = {v: e for e, v in enumerate(list(vocab))}
matrix = np.zeros((len(docs), len(vocab)))


rows = []
cols = []
data = []
for e, d in tqdm(enumerate(docs)):
    any = False
    for w in d:
        if w in v2id:
            rows.append(d2id[ids[e]])
            cols.append(v2id[w])
            data.append(d[w])
            any = True
    if not any:
        print(d)

from scipy.sparse import csc_array
idx = csc_array((data, (rows, cols)), shape=(len(docs), len(vocab)))