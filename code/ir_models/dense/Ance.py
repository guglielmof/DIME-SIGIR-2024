from .AbstractSentenceTransformer import AbstractSentenceTransformer


class Ance(AbstractSentenceTransformer):

    def __init__(self, model_hgf="sentence-transformers/msmarco-roberta-base-ance-firstp"):
        super().__init__(model_hgf=model_hgf)

        self.name = "ance"
        self.embeddings_dim = 768
