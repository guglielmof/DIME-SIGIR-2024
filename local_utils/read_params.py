import itertools


def read_params(predictor, configs):

    types = {"int": int}

    if  configs.get(f"{predictor}.params") is None:
        paramset  = []
    else:
        params = configs.get(f"{predictor}.params").split(",")
        paramset = {}
        for p in params:
            vals = configs.get(f"{predictor}.{p}").split(",")
            if not configs.get(f"{predictor}.{p}.type") is None and configs.get(f"{predictor}.{p}.type") in types:
                vals = list(map(types[configs.get(f"{predictor}.{p}.type")], vals))
            paramset[p] = vals

        paramset = [dict(zip(paramset, x)) for x in itertools.product(*paramset.values())]

    return paramset

