import faiss

coll2corpus = {'deeplearning19': 'msmarco-passages',
               'deeplearning20': 'msmarco-passages',
               'deeplearninghd': 'msmarco-passages',
               'robust04': 'tipster'}


def load_index(index_dir, collection, model_name):
    corpus = coll2corpus[collection]
    index_path = f"{index_dir}/{corpus}/{model_name}"
    index_filename = f"{index_path}/{model_name}.faiss"
    # Load the index from the file
    index = faiss.read_index(index_filename)

    mapper = list(map(lambda x: x.strip(), open(f"{index_path}/{model_name}.map", "r").readlines()))

    return index, mapper
