import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class MemmapEncoding:

    def __init__(self, datapath, mappingpath, embedding_size=768, index_name="id", sep=","):
        self.data = np.memmap(datapath, dtype=np.float32, mode="r").reshape(-1, embedding_size)
        self.id2int = {}

        with open(mappingpath, "r") as fp:
            for l in fp.readlines():
                idx, offset = l.strip().split(sep)
                if offset.isnumeric():
                    self.id2int[idx] = int(offset)

        self.int2id = {v: k for k, v in self.id2int.items()}
        self.shape = self.get_shape()

    def get_ids(self):
        return list(self.id2int.keys())

    def get_shape(self):
        return self.data.shape

    def get_encoding(self, idx):
        return self.data[self.get_position(idx)]

    def get_data(self):
        return self.data

    def get_position(self, idx):
        if type(idx) == list or type(idx) == np.array:
            return [self.id2int[i] for i in idx]
        else:
            return self.id2int[idx]