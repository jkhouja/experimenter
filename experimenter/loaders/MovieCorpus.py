import codecs
import os
import re


class MovieCorpus:
    """A class to load cornel movie corpus data"""

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
