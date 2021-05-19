import logging
import os
import re


class DialogueTextFile:
    def __init__(
        self,
        data_path,
        input_paths,
        label_name,
        eoc_regex,
        eoc,
        limit=None,
    ):
        self.input_paths = [
            os.path.join(data_path, input_path) for input_path in input_paths
        ]
        self.label_name = label_name
        self.eoc_regex = eoc_regex
        self.eoc = eoc
        self.limit = limit

    def load_dialogue(self, path: str):

        splitter = f"(.*){self.eoc_regex}(.*)"
        with open(path, "r") as f:
            lines = f.readlines()

        res = []
        for i, line in enumerate(lines):
            if self.limit and i == self.limit:
                break
            match = re.findall(splitter, line)
            if match is None or len(match) == 0 or len(match[0]) != 2:
                logging.warning(
                    f"Could not split line {i} into context/response pair.\
                        got {len(match)} splits: \n"
                )
                logging.warning(line)
            else:
                match = match[0]
                res.append(
                    [match[0].strip() + " " + self.eoc, {self.label_name: match[1].strip()}]
                )
        print("Loading data from disk finished")
        return res

    def __call__(self):
        out = []
        for path in self.input_paths:
            out.append(self.load_dialogue(path))
        return out
