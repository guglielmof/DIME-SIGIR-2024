from transformers import AutoTokenizer, AutoModel
from .AbstractDenseModel import AbstractDenseModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np

class generic_starbucks(nn.Module):
    def __init__(self, model_name, sizes):
        super().__init__()
        self.sizes = sizes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.mask = np.ones((768, len(self.sizes)))
        for e, i in enumerate(self.sizes):
            self.mask[i[1]:, e] = 0

    def tokenize(self, input_text):
        return self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    def forward(self, inps):
        outputs = self.model(**inps, return_dict=True, output_hidden_states=True)
        encodings = torch.stack([outputs.hidden_states[l][:, 0, :] for l, _ in self.sizes], dim=2)
        query_encodings = np.multiply(encodings.to("cpu"), self.mask[np.newaxis, :, :])
        return {"sentence_embedding": query_encodings}



class Starbucks(AbstractDenseModel):

    def __init__(self, *args, model_hgf='ielabgroup/Starbucks-msmarco', **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf
        self.embeddings_dim = 768

        self.sizes = kwargs.get('sizes', [(2, 32), (4, 64), (6, 128), (8, 256), (10, 512), (12, 768)])

        docs_model = generic_starbucks(model_hgf, self.sizes)

        self.docs_model = SentenceTransformer(modules=[docs_model])

        query_model = generic_starbucks(model_hgf, self.sizes)
        self.queries_model = SentenceTransformer(modules=[query_model])


        self.name = "Starbucks"

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

