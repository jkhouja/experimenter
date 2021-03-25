import codecs
import os
import re

import pandas as pd
from nltk import CFG
from nltk.parse.generate import generate
from torchnlp.datasets import imdb_dataset


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


class LMCSV:
    """A class that takes a csv file (usually saved by pandas)
    and a inp_col name and create a LM tuple record:
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
        # self.logger.info("All loaded data size:{}".format(data_in.shape[0]))

        def f(x):
            return self.beg_sym + x + self.end_sym

        out = []
        for split in data_in:
            s1_data = [f(x[self.inp_col]) for x in split.to_dict(orient="records")]
            out.append(
                [
                    (
                        [d[: -(len(self.end_sym))]],
                        {self.label_name: d[len(self.beg_sym) :]},  # noqa: E203
                    )
                    for d in s1_data
                ]
            )
        return out


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
                    ([x[self.inp_col]], {self.label_name: [x[self.out_col]]})
                    for x in d.to_dict(orient="records")
                ]
            )
        return out


class MovieCorpus:
    """A class to load cornel movie corpus data (train only)"""

    def __init__(self, data_path, input_paths, label_col, limit=None):
        self.paths = [os.path.join(data_path, input_path) for input_path in input_paths]
        self.label_col = label_col
        self.limit = limit

    def __call__(self):
        # Define path to new file
        # datafile = os.path.join(self.path, "formatted_movie_lines.txt")

        delimiter = "\t"
        # Unescape the delimiter
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))

        # Initialize lines dict, conversations list, and field ids
        lines = {}
        conversations = []
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
        MOVIE_CONVERSATIONS_FIELDS = [
            "character1ID",
            "character2ID",
            "movieID",
            "utteranceIDs",
        ]

        # Load lines and process conversations
        print("\nProcessing corpus...")
        lines = self.loadLines(
            os.path.join(self.paths[0], "movie_lines.txt"), MOVIE_LINES_FIELDS
        )
        print("\nLoading conversations...")
        conversations = self.loadConversations(
            os.path.join(self.paths[0], "movie_conversations.txt"),
            lines,
            MOVIE_CONVERSATIONS_FIELDS,
        )

        res = []
        for pair in self.extractSentencePairs(conversations):
            res.append([pair[0], {self.label_col: pair[1]}])

        print("Loading data from disk finished")
        if self.limit:
            return [res[: self.limit]]
        return [res]

    # Splits each line of the file into a dictionary of fields
    def loadLines(self, fileName, fields):
        lines = {}
        with open(fileName, "r", encoding="iso-8859-1") as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj["lineID"]] = lineObj
        return lines

    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    def loadConversations(self, fileName, lines, fields):
        conversations = []
        with open(fileName, "r", encoding="iso-8859-1") as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                utterance_id_pattern = re.compile("L[0-9]+")
                lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)
        return conversations

    # Extracts pairs of sentences from conversations
    def extractSentencePairs(self, conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(
                len(conversation["lines"]) - 1
            ):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i + 1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs


class ClassCFG:
    """A class that generates synthetic data for a classification task record:
    (sentence, {label_name: class})
    """

    def __init__(self, label_name, data_path, limit=None):
        self.label_name = label_name
        self.limit = limit
        sent_prod_grammar = """
            S -> IDENT POS_STATE '+' | IDENT NEG_STATE '-'
            IDENT -> 'it' | Det_sing PROD | Det_sing PROD CONNECT PROD_IDENT |\
                    Det_sing DESC POINTER Det_sing PROD | Det_sing PROD DESC |\
                    Det_sing DESC POINTER Det_sing PROD CONNECT PROD_IDENT
            CONNECT -> 'which' | 'that'
            PROD_IDENT -> N VP_PAST
            PROD -> PROD_NAME
            DESC -> DESC_NAME
            POINTER -> 'of'
            POS_STATE -> 'is' POSITIVE | 'is' AUX POSITIVE |\
                    'in my opinion is' POSITIVE | 'cannot be found anywhere else'
            NEG_STATE -> 'is' NEGATIVE | 'is' AUX NEGATIVE |\
                    'in my opinion is' NEGATIVE | 'is a waste of money!'
            AUX -> 'honestly|totally'
            POSITIVE -> 'great'
            NEGATIVE -> 'bad'
            Det_sing -> 'the' | 'this' | 'that'
            Det_plur -> 'these' | 'the' | 'the {22}'
            UnDet_sing -> 'a' | 'such'
            UnDet_sing_vowel -> 'an'
            PROD_NAME -> 'PS5' | 'Instagram' | 'camera' | 'DSLR'
            DESC_NAME -> '{product_descriptor}'
            N -> '{group_of_people}'
            VP_PAST -> '{purchased}'
            """
        self.grammar = CFG.fromstring(sent_prod_grammar)

    def __call__(self):
        out = []
        for sentence in generate(self.grammar, n=self.limit):
            text = " ".join(sentence[:-1])
            label = sentence[-1]
            out.append((text, {self.label_name: label}))

        return [out]


class ClassIMDB:
    def __init__(
        self, train=True, test=False, data_path="imdg_data/", label_name="sentiment"
    ):
        self.train = train
        self.test = test
        self.directory = data_path
        self.label_name = label_name

    def process_fold(self, fold: list) -> list:
        res = []
        for record in fold:
            res.append((record["text"], {self.label_name: [record["sentiment"]]}))
        return res

    def __call__(self):

        folds = imdb_dataset(train=self.train, test=self.test, directory=self.directory)
        out = []
        if self.train and self.test:
            out.append(self.process_fold(folds[0]))
            out.append(self.process_fold(folds[1]))

        else:
            out.append(self.process_fold(folds))

        return out
