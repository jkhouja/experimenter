
import importlib

def is_str(func):
    """Decorator that checks that attribute type is a string
    """
    def decorated(input_text: str, **kwargs):
        if not isinstance(input_text, str):
            raise AttributeError("Expecting a string, but got a {}".format(type(input_text)))
        else:
            return func(input_text, **kwargs)
    return decorated

def list_applyer(func):
    def decorated(self, input_data, list_input=False, **kwargs):
        if not list_input:
            return func(self, input_data, **kwargs)
        else:
            res = []
            for inp in input_data:
                #Build sequence of characters and truncate padding
                res.append(func(self, inp, **kwargs))
            return res
    return decorated

class chainer:
    def __init__(self, funcs):
        self._funcs = funcs
    
    @list_applyer
    def __call__(self, data):
        for func in self._funcs:
            data = func(data)
        return data

def load_class(module_name, class_name, class_param,  pass_params_as_dict=False):
    module = importlib.import_module(module_name)
    class_n = getattr(module, class_name)
    if pass_params_as_dict:
        return(class_n(**class_param))
    return class_n(class_param)

def evaluate_params(params, local_vars=None):
    locals().update(local_vars)
    for p in params:
        if isinstance(params[p], dict) and params[p]['eval']:
            params[p] = eval(params[p]['value'])
