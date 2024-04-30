import pandas as pd

import ir_measures


def compute_measure(run, qrels, measure_name):
    measure = [ir_measures.parse_measure(measure_name)]
    out = pd.DataFrame(ir_measures.iter_calc(measure, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out

