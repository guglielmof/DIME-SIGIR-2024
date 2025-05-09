from transformers import AutoTokenizer, AutoModel
from .AbstractDenseModel import AbstractDenseModel
from sentence_transformers import SentenceTransformer
import torch.nn as nn

from collections import namedtuple


class AbstractDragonTorch(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name.query)

    def forward(self, inps):
        return {"sentence_embedding": self.encoder(**inps).last_hidden_state[:, 0, :]}


class DocumentsDragonTorch(AbstractDragonTorch):

    def __init__(self, model_name):
        super().__init__(model_name)
        self.encoder = AutoModel.from_pretrained(model_name.context)

    def tokenize(self, input_text):
        return self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt', max_length=512)


class QueriesDragonTorch(AbstractDragonTorch):

    def __init__(self, model_name):
        super().__init__(model_name)
        self.encoder = AutoModel.from_pretrained(model_name.query)

    def tokenize(self, input_text):
        return self.tokenizer(input_text,  return_tensors='pt', padding=True, truncation=True, max_length=36)


class Dragon(AbstractDenseModel):

    def __init__(self, *args, model_hgf='default', **kwargs):
        super().__init__(*args, **kwargs)
        if model_hgf == "default":
            DragonNames = namedtuple("DragonsHgfPointers", "query context")
            model_hgf = DragonNames('facebook/dragon-plus-query-encoder', 'facebook/dragon-plus-context-encoder')

        self.model_hgf = model_hgf

        docs_model = DocumentsDragonTorch(model_hgf)

        self.docs_model = SentenceTransformer(modules=[docs_model])

        query_model = QueriesDragonTorch(model_hgf)
        self.queries_model = SentenceTransformer(modules=[query_model])

        self.embeddings_dim = 768

        self.name = "dragon"

    def encode_queries(self, texts):
        return self.queries_model.encode(texts)

    def encode_documents(self, texts):
        return self.docs_model.encode(texts)

    def start_multi_process_pool(self):
        return self.docs_model.start_multi_process_pool()

    def stop_multi_process_pool(self, pool):
        self.docs_model.stop_multi_process_pool(pool)

    def get_model(self):
        return self.docs_model
