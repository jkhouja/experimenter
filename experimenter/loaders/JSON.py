import os

import pandas


class LMJson:
    def __init__(
        self,
        data_path,
        input_paths,
        label_name,
        input_names,
        join_sym="<sep>",
        include_empty_lines=False,
        beg_sym="<s> ",
        end_sym=" <e>",
        limit=None,
    ):
        self.input_paths = [
            os.path.join(data_path, input_path) for input_path in input_paths
        ]
        self.label_name = label_name
        self.input_names = input_names
        self.join_sym = join_sym
        self.inlclude_empty_lines = include_empty_lines
        self.beg_sym = beg_sym
        self.end_sym = end_sym
        self.limit = limit

    def __call__(self):
        out = []
        for path in self.input_paths:
            data = []
            split = pandas.read_json(path, lines=True, nrows=self.limit)

            for index, row in split.iterrows():
                sent = self.join_sym.join([row[a] for a in self.input_names])
                if sent == "" and self.include_empty_lines:
                    continue

                tmp = (
                    self.beg_sym + sent,
                    {self.label_name: sent + self.end_sym},
                )
                data.append(tmp)
            out.append(data)

        return out


class ClassJson:
    def __init__(
        self,
        data_path,
        input_paths,
        input_names,
        label_cols,
        label_names,
        join_sym=" ",
        include_empty_lines=False,
        limit=None,
    ):
        self.input_paths = [
            os.path.join(data_path, input_path) for input_path in input_paths
        ]

        if len(label_cols) != len(label_names):
            raise AttributeError(
                "Label names to override should be the same size as label_cols"
            )

        self.label_cols = label_cols
        self.label_names = label_names
        self.input_names = input_names
        self.join_sym = join_sym
        self.include_empty_lines = include_empty_lines
        self.limit = limit

    def __call__(self):
        out = []
        for path in self.input_paths:
            data = []
            split = pandas.read_json(path, lines=True)

            nrows = 0
            for index, row in split.iterrows():
                if self.limit and nrows == self.limit:
                    break
                sent = self.join_sym.join([row[a] for a in self.input_names])
                if sent == "" and not self.include_empty_lines:
                    continue

                tmp = (
                    sent,
                    {
                        l: [row[l_c]]
                        for l, l_c in zip(self.label_names, self.label_cols)
                    },
                )
                data.append(tmp)
                nrows += 1
            out.append(data)

        return out
