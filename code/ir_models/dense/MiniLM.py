from .AbstractSentenceTransformer import AbstractSentenceTransformer


class MiniLM(AbstractSentenceTransformer):

    def __init__(self, model_hgf='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__(model_hgf=model_hgf)

        self.name = "minilm"