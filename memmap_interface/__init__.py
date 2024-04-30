import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class AbsractMemmapEncoding:

    def __init__(self, datapath, mappingpath, embedding_size=768, index_name="id", sep=","):
        self.data = np.memmap(datapath, dtype=np.float32, mode="r").reshape(-1, embedding_size)
        self.mapping = pd.read_csv(mappingpath, dtype={index_name: str}, sep=sep).set_index(index_name)

        self.shape = self.get_shape()

    def get_position(self, idx):
        return self.mapping.loc[idx]

    def get_inverse_position(self, offset):
        return self.mapping.loc[self.mapping["offset"]==offset].index[0]


    def get_encoding(self, idx):
        return self.data[self.mapping.loc[idx, "offset"]]

    def get_centroid(self):
        if not hasattr(self, "centroid"):
            self.centroid = np.mean(self.data, axis=0)
        return self.centroid

    def normalize_data(self):
        if not hasattr(self, "normalized_data"):
            self.normalized_data = normalize(self.data)

    def get_normalized_encoding(self, idx):
        self.normalize_data()
        return self.normalized_data[self.mapping.loc[idx, "offset"]]

    def get_data(self, normalized=False):
        if normalized:
            self.normalize_data()
            return self.normalized_data
        else:
            return self.data

    def get_shape(self):

        return self.data.shape

    def get_ids(self):
        return self.mapping.index

class MemmapCorpusEncoding(AbsractMemmapEncoding):

    def __init__(self, datapath, mappingpath, embedding_size=768):
        super().__init__(datapath, mappingpath, embedding_size, index_name="doc_id")


class MemmapQueriesEncoding(AbsractMemmapEncoding):
    def __init__(self, datapath, mappingpath, embedding_size=768):
        super().__init__(datapath, mappingpath, embedding_size, index_name="qid", sep="\t")
        self.data = self.data[self.mapping.offset, :]
        self.mapping.offset = np.arange(len(self.mapping.index))
