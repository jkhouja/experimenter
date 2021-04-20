import os

import pandas as pd


class LMCSV:
    """A class that takes a csv file (usually saved by pandas) and
    a inp_col name and create a LM tuple record:
    (sentence, {label_name: sentence_out})
    after adding beg_sym and end_sym"""

    def __init__(
        self,
        data_path,
        input_paths,
        inp_col,
        label_name,
        beg_sym="<s> ",
        end_sym=" <e>",
        limit=None,
    ):
        self.input_paths = [
            os.path.join(data_path, input_path) for input_path in input_paths
        ]
        self.inp_col = inp_col
        self.label_name = label_name
        self.beg_sym = beg_sym
        self.end_sym = end_sym
        self.limit = limit

    def __call__(self):
        data_in = []
        for path in self.input_paths:

            data_in.append(pd.read_csv(path))

        if self.limit is not None and self.limit > 0:
            # Limit the first split (train) size
            data_in[0] = data_in[0][: self.limit]
        # data_in['stance'] = data_in['stance'].astype(str)

        def f(x):
            return self.beg_sym + x + self.end_sym

        out = []
        for split in data_in:
            s1_data = [f(x[self.inp_col]) for x in split.to_dict(orient="records")]
            print(s1_data)
            out.append(
                [
                    (
                        d[: -(len(self.end_sym))],
                        {self.label_name: d[len(self.beg_sym) :]},  # noqa: E203
                    )
                    for d in s1_data
                ]
            )
        return out
