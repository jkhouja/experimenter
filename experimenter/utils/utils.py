import importlib
import json


def safe_serialize(obj):
    def format_obj_type(field: object) -> str:
        return f"<<non-serializable: {type(field).__qualname__}>>"

    return json.dumps(obj, default=format_obj_type)


def is_str(func):
    """Decorator that checks that attribute type is a string"""

    def decorated(input_text: str, **kwargs):
        if not isinstance(input_text, str):
            raise AttributeError(
                "Expecting a string, but got a {}".format(type(input_text))
            )
        return func(input_text, **kwargs)

    return decorated


def list_applyer_old(func):
    """Decorator that applies the decorated function to each item of input_data"""

    def decorated(self, input_data, list_input=False, **kwargs):
        if not list_input:
            return func(self, input_data, **kwargs)
        res = []
        for inp in input_data:
            # Build sequence of characters and truncate padding
            res.append(func(self, inp, **kwargs))
        return res

    return decorated


def list_applyer(func):
    """Decorator that applies the decorated function to each item of input_data"""

    def decorated(self, input_data, list_input=False, **kwargs):
        if not list_input:
            return func(self, input_data, **kwargs)
        else:
            return func(self, input_data, list_input=True, **kwargs)

    return decorated


class chainer:
    """Chain applies the functions in sequence"""

    def __init__(self, funcs):
        self._funcs = funcs

    # @list_applyer
    def __call__(self, data, list_input=False):
        for func in self._funcs:
            data = func(data, list_input)
        return data


def move_batch(batch, device):
    """Moves a batch of dictionary to the corresponding local device"""
    res = {}
    for k in batch.keys():
        if isinstance(batch[k], list):
            res[k] = [i.to(device) for i in batch[k]]
        else:
            res[k] = batch[k].to(device)
    return res


def load_class(module_name, class_name, class_param, pass_params_as_dict=False):
    """Loads a class given it's module name, class name, and dict of params"""
    module = importlib.import_module(module_name)
    class_n = getattr(module, class_name)
    if pass_params_as_dict:
        return class_n(**class_param)
    return class_n(class_param)


def evaluate_params(params, local_vars=None):
    """Helper function that dynamically evaluates parameters
    that have been set by previous steps"""
    locals().update(local_vars)
    try:
        if isinstance(params, dict):
            for p in params:
                if isinstance(params[p], dict) and params[p]["eval"]:
                    params[p] = eval(params[p]["value"])
                elif isinstance(params[p], list):
                    evaluate_params(params[p], locals())
        elif isinstance(params, list):
            for p in params:
                if isinstance(p, dict) and p["eval"]:
                    print("Should be here!")
                    p = eval(p["value"])
                    print(p)
                elif isinstance(p, list):
                    evaluate_params(p, locals())

    except Exception as e:
        print("couldn't evaluate parameter: {}".format(p))
        raise e
