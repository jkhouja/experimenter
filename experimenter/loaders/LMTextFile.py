import os


class LMTextFile:
    def __init__(
        self,
        data_path,
        input_paths,
        label_name,
        break_line_at,
        beg_sym="<s> ",
        end_sym=" <e>",
        limit=None,
    ):
        self.input_paths = [
            os.path.join(data_path, input_path) for input_path in input_paths
        ]
        self.label_name = label_name
        self.beg_sym = beg_sym
        self.end_sym = end_sym
        self.limit = limit
        self.seq_len = break_line_at

    def __call__(self):
        out = []
        for path in self.input_paths:
            data = []
            with open(path, "r") as f:
                word_in_line = 0
                sent = []
                for i, line in enumerate(f.readlines()):
                    if i == self.limit:
                        break
                    if line.strip() != "":
                        for word in line.strip().split():
                            sent.append(word)
                            word_in_line += 1
                            if (
                                self.seq_len is not None
                                and word_in_line == self.seq_len - 1
                            ):
                                # We reached the end of sequence, add to output and restart
                                tmp = (
                                    [self.beg_sym + " ".join(sent)],
                                    {self.label_name: " ".join(sent) + self.end_sym},
                                )
                                data.append(tmp)
                                sent = []
                                word_in_line = 0

                out.append(data)

        return out
