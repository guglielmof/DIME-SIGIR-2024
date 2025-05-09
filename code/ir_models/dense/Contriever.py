from .AbstractSentenceTransformer import AbstractSentenceTransformer


class Contriever(AbstractSentenceTransformer):

    def __init__(self, model_hgf="facebook/contriever-msmarco"):
        super().__init__(model_hgf=model_hgf)

        self.name = "contriever"
        self.embeddings_dim = 768