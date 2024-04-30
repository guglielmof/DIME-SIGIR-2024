from .get_configs import get_configs
from .read_params import read_params
import math

def chunk_based_on_number(lst, chunk_numbers):
    n = math.ceil(len(lst) / chunk_numbers)

    chunks = []
    for x in range(0, len(lst), n):
        each_chunk = lst[x: n + x]
        chunks.append(each_chunk)

    return chunks
