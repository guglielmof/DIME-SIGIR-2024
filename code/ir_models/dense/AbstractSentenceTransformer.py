from sentence_transformers import SentenceTransformer
from .AbstractDenseModel import AbstractDenseModel


class AbstractSentenceTransformer(AbstractDenseModel):

    def __init__(self, *args, model_hgf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf
        self.model = SentenceTransformer(model_hgf)
        self.embeddings_dim = self.model[1].word_embedding_dimension

    def encode_queries(self, texts):
        return self.model.encode(texts)

    def encode_documents(self, texts):
        return self.model.encode(texts)

    def start_multi_process_pool(self):
        return self.model.start_multi_process_pool()

    def stop_multi_process_pool(self, pool):
        self.model.stop_multi_process_pool(pool)

    def get_model(self):
        return self.model
