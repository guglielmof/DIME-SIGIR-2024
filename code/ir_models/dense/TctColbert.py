from transformers import AutoTokenizer, AutoModel
from .AbstractDenseModel import AbstractDenseModel
from sentence_transformers import SentenceTransformer
import torch.nn as nn


class generic_tctcolbert(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


class docs_tctcolbert(generic_tctcolbert):

    def __init__(self, model_name):
        super().__init__(model_name)

    def tokenize(self, input_text):
        inps = self.tokenizer([f'[CLS] [D] {d}' for d in input_text], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True,
                              max_length=512)

        return inps

    def forward(self, inps):
        inps = {k: v for k, v in inps.items()}
        res = self.model(**inps).last_hidden_state
        res = res[:, 4:, :]  # remove the first 4 tokens (representing [CLS] [ D ])
        res = res * inps['attention_mask'][:, 4:].unsqueeze(2)  # apply attention mask
        lens = inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1)
        lens[lens == 0] = 1  # avoid edge case of div0 errors
        res = res.sum(dim=1) / lens  # average based on dim
        # print(res.cpu().numpy())
        # print(res)
        return {"sentence_embedding": res}


class queries_tctcolbert(generic_tctcolbert):

    def __init__(self, model_name):
        super().__init__(model_name)

    def tokenize(self, input_text):
        inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in input_text], add_special_tokens=False, return_tensors='pt',
                              padding=True, truncation=True, max_length=36)
        #inps = self.tokenizer([q for q in input_text], add_special_tokens=False, return_tensors='pt',
        #                      padding=True, truncation=True, max_length=36)

        return inps

    def forward(self, inps):
        inps = {k: v for k, v in inps.items()}
        res = self.model(**inps).last_hidden_state
        res = res[:, 4:, :].mean(dim=1)  # remove the first 4 tokens (representing [CLS] [ Q ]), and average

        return {"sentence_embedding": res}


class TctColbert(AbstractDenseModel):

    def __init__(self, *args, model_hgf='castorini/tct_colbert-v2-msmarco', **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf

        docs_model = docs_tctcolbert(model_hgf)

        self.docs_model = SentenceTransformer(modules=[docs_model])

        query_model = queries_tctcolbert(model_hgf)
        self.queries_model = SentenceTransformer(modules=[query_model])

        self.embeddings_dim = 768

        self.name = "tctcolbert"

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


