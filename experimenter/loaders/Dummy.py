class Dummy:
    """Dummy data generator that generates LM data based on 2 simple sentences and 2 classes"""

    def __init__(self, data_path, data_size, lm_label, cls_label):
        self.beg_sym = "<s> "
        self.end_sym = " <e>"
        self.lm_label = lm_label
        self.cls_label = cls_label
        self.data_size = data_size

    def __call__(self):
        out = []
        # Each record is a touple of [sentence] , {label1: [labels], label2: [labels]})
        s = "one two three four"
        b = "five six seven eight nine"

        for i in range(self.data_size // 2):
            out.append(
                (
                    self.beg_sym + s,
                    {self.lm_label: s + self.end_sym, self.cls_label: [0]},
                )
            )
            out.append(
                (
                    self.beg_sym + b,
                    {self.lm_label: b + self.end_sym, self.cls_label: [1]},
                )
            )

        return [out]
