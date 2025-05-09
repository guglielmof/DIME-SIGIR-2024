from .AbstractSentenceTransformer import AbstractSentenceTransformer


class Tasb(AbstractSentenceTransformer):

    def __init__(self, model_hgf="sentence-transformers/msmarco-distilbert-base-tas-b"):
        super().__init__(model_hgf=model_hgf)

        self.name = "tasb"
        self.embeddings_size = 768