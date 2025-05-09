class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractDenseModel(metaclass=Singleton):

    def __init__(self, *args, **kwargs):
        self.name = "unnamed"
        self.embeddings_dim = None

    def encode_queries(self, texts):
        raise NotImplementedError

    def encode_documents(self, texts):
        raise NotImplementedError

    def get_name(self):
        return self.name

    def get_embedding_dim(self):
        return self.embeddings_dim
