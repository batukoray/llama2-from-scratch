from typing import List


class SimpleTokenizer:
    """
    Minimal word-level tokenizer implementation.
    """

    __vocabulary: dict[str, int]
    __reverse_vocabulary: dict[int, str]
    __unk_token: str
    __bos_token: str
    __eos_token: str

    def __init__(self):
        """
        Initializes the tokenizer with special tokens.
        """
        self.__unk_token = "<unk>"
        self.__bos_token = "<bos>"
        self.__eos_token = "<eos>"

        self.__vocabulary = {
            self.__unk_token: 0,
            self.__bos_token: 1,
            self.__eos_token: 2
        }
        self.__reverse_vocabulary = {
            0: self.__unk_token,
            1: self.__bos_token,
            2: self.__eos_token
        }

    def fit(self, text: str) -> None:
        """
        Learns vocabulary entries from whitespace-separated text.
        """
        tokens = text.split()

        for token in tokens:
            if token not in self.__vocabulary:
                index = len(self.__vocabulary)
                self.__vocabulary[token] = index
                self.__reverse_vocabulary[index] = token

    def encode(self, text: str) -> List[int]:
        """
        Converts whitespace-separated text into token ids.
        """
        tokens = text.split()
        unknown_id = self.__vocabulary[self.__unk_token]

        ids = []

        for token in tokens:
            ids.append(self.__vocabulary.get(token, unknown_id))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Converts token ids back into a whitespace-separated string.
        """
        tokens = []

        for token_id in ids:
            tokens.append(self.__reverse_vocabulary.get(token_id, self.__unk_token))

        return " ".join(tokens)

    def getVocabularySize(self) -> int:
        """
        Returns the current vocabulary size.
        """
        return len(self.__vocabulary)

    def getBosId(self) -> int:
        """
        Returns the id of the beginning-of-sequence token.
        """
        return self.__vocabulary[self.__bos_token]

    def getEosId(self) -> int:
        """
        Returns the id of the end-of-sequence token.
        """
        return self.__vocabulary[self.__eos_token]

    def getUnkId(self) -> int:
        """
        Returns the id of the unknown token.
        """
        return self.__vocabulary[self.__unk_token]
