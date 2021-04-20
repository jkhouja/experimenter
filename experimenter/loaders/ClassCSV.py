import os

import pandas as pd


class ClassCSV:
    """A class that takes a csv file (usually saved by pandas)
    and an inp_col / out_col and create a classification record:
    (sentence, {label_name: class})
    """

    def __init__(
        self, data_path, input_paths, inp_col, out_col, label_name, limit=None
    ):
        self.input_paths = [
            os.path.join(data_path, input_path) for input_path in input_paths
        ]
        self.inp_col = inp_col
        self.out_col = out_col
        self.label_name = label_name
        self.limit = limit

    def __call__(self):
        data_in = []
        for path in self.input_paths:

            data_in.append(pd.read_csv(path))

        if self.limit is not None and self.limit > 0:
            # Limit the first split (train) size
            data_in[0] = data_in[0][: self.limit]

        out = []
        for d in data_in:
            d[self.out_col] = d[self.out_col].astype(str)
            # self.logger.info("All loaded data size:{}".format(data_in.shape[0]))

            out.append(
                [
                    (x[self.inp_col], {self.label_name: [x[self.out_col]]})
                    for x in d.to_dict(orient="records")
                ]
            )
        return out
