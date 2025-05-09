from .AbstractSentenceTransformer import AbstractSentenceTransformer


class Minilml12(AbstractSentenceTransformer):

    def __init__(self, model_hgf='sentence-transformers/all-MiniLM-L12-v2'):
        super().__init__(model_hgf=model_hgf)

        self.name = "minilml12"
        self.embeddings_dim = 384
