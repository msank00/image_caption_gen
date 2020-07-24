import os.path
import pickle
from collections import Counter

import nltk
from tqdm import tqdm

from src.utils import get_training_data


class Vocabulary:
    def __init__(
        self,
        vocab_threshold: int,
        vocab_file: str,
        caption_file: str,
        image_id_file: str,
        vocab_from_file: str,
        start_word: str = "<start>",
        end_word: str = "<end>",
        unk_word="<unk>",
    ):

        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          caption_file: Path for caption file
          image_id_file: Path for training id file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """

        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.caption_file = caption_file
        self.image_id_file = image_id_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_from_file = vocab_from_file
        self._get_vocab()

    def _get_vocab(self):
        """Build vocabulary 1. from scratch or 2. from external file
        """

        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, "rb") as fin:
                vocab = pickle.load(fin)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
            print(
                f"Vocabulary successfully loaded from external file {self.vocab_file}..."
            )
        else:
            self._build_vocab()
            with open(self.vocab_file, "wb") as fout:
                pickle.dump(self, fout)
            print(
                f"Vocabulary created from scratch and saved to pickle file {self.vocab_file}..."
            )

    def _build_vocab(self):
        """Build vocabulary dictionary.
            1. index2word
            2. word2index
        """
        self._init_vocab()
        for base_vocab in [self.start_word, self.end_word, self.unk_word]:
            self._add_word(base_vocab)

        self._build_vocab_from_caption()

    def _init_vocab(self):
        """Initialize the vocabulary
        """
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def _add_word(self, word: str):
        """Add token to the vocabulary dictionary

        :param word: tokens to add
        :type word: str
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def _build_vocab_from_caption(self):
        """Loop over training caption and add all those tokens
        to the vocabulary whose token count >= vocab_threshold
        """

        df_train = get_training_data(self.image_id_file, self.caption_file)

        all_captions = df_train.CAPTION.values.tolist()
        counter = Counter()

        for i, caption in tqdm(
            enumerate(all_captions), total=len(all_captions)
        ):
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

        words = [
            word
            for word, cnt in counter.items()
            if cnt >= self.vocab_threshold
        ]

        for i, word in tqdm(enumerate(words), total=len(words)):
            self._add_word(word)

    def __call__(self, word: str):
        """Return index of the word

        :param word: Index to find for
        :type word: str
        :return: Index of the word
        :rtype: int
        """
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
