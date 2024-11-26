from torch.nn import functional as F
from torch.utils.data import Dataset
import torch
import nltk
from nltk.corpus import conll2000


# TODO: implement losses?
# TODO: implement datasets and dataloaders
# TODO: implement MDL calculation
# TODO: implement metrics


class POSTaggingTokenizer:
    """
    Tokenizer for part-of-speech tagging.

    This tokenizer converts words and their part-of-speech tags into IDs and
    provides functionality for encoding and decoding tags.

    Attributes:
        idx2tag (dict): Maps tag IDs to tags.
        tag2idx (dict): Maps tags to tag IDs.
        pad_token_id (int): ID for the padding token.
        bos_token_id (int): ID for the beginning-of-sequence token.
        eos_token_id (int): ID for the end-of-sequence token.
        unk_token_id (int): ID for the unknown token.
    """

    def __init__(self):
        """
        Initializes the POSTaggingTokenizer with special tokens for padding,
        beginning-of-sequence, end-of-sequence, and unknown tokens.
        """
        self.idx2tag = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.tag2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.idx2tag)

    def fit(self, sentences_conll):
        """
        Builds the tag vocabulary from a list of sentences in the CoNLL format.

        Args:
            sentences_conll (List[List[tuple]]): List of sentences, where each
                sentence is a list of (word, tag) tuples.

        Returns:
            POSTaggingTokenizer: The fitted tokenizer.
        """
        idx = len(self.idx2tag)
        for s in sentences_conll:
            for _, word_tag in s:
                if self.tag2idx.get(word_tag) is None:
                    self.tag2idx[word_tag] = idx
                    self.idx2tag[idx] = word_tag
                    idx += 1
        return self

    def encode(self, sent_words_with_tags):
        """
        Encodes a list of words and their tags into IDs.

        Args:
            sent_words_with_tags (List[List[tuple]] or List[tuple]): List of sentences
                or a single sentence, where each sentence is represented as a list
                of (word, tag) tuples.

        Returns:
            List[List[tuple]] or List[tuple]: Encoded sentences or a single sentence
            with tags replaced by their IDs.
        """
        if isinstance(sent_words_with_tags[0], tuple):
            sent_words_with_tags = [sent_words_with_tags]
        sents_w_with_ids = []
        for s in sent_words_with_tags:
            w_with_ids = []
            for w, t in s:
                if self.tag2idx.get(t) is not None:
                    w_with_ids.append((w, self.tag2idx[t]))
                else:
                    w_with_ids.append((w, self.tag2idx["<UNK>"]))
            sents_w_with_ids.append(w_with_ids)

        if len(sents_w_with_ids) == 1:
            sents_w_with_ids = sents_w_with_ids[0]

        return sents_w_with_ids

    def __call__(self, NERs):
        """
        see .encode(...)
        """
        return self.encode(NERs)

    def decode(self, ids):
        """
        Decodes a list of tag IDs into part-of-speech tags.

        Args:
            ids (list of int): List of tag IDs.

        Returns:
            list: List of decoded tags.
        """
        tags = []
        for idx in ids:
            if self.idx2tag.get(idx) is not None:
                tags.append(self.idx2tag[idx])
            else:
                tags.append(self.unk_token_id)
        return tags


class MDLDataset_POSTagging(Dataset):
    """
    Dataset class for part-of-speech tagging with MDL.

    Args:
        data_path (str): Path to the dataset file. Must be "conll2000_train.txt" or "conll2000_test.txt".
        tagging_tokenizer (POSTaggingTokenizer, optional): Tokenizer for POS tagging.
            If None, a new tokenizer will be created and fitted to the data.

    Attributes:
        tagging_tokenizer (POSTaggingTokenizer): Tokenizer for encoding tags.
        data (list): List of encoded sentences with words and tag IDs.
    """

    def __init__(self, data_path, tagging_tokenizer=None):
        """
        Initializes the MDLDataset_POSTagging class.
        """
        super().__init__()
        self.tagging_tokenizer = tagging_tokenizer
        self.data = self._read_data(data_path)

    def _read_data(self, data_path):
        """
        Reads and processes data from the specified path.

        Args:
            data_path (str): Path to the dataset file.

        Returns:
            list: List of sentences with words and tag IDs.

        Raises:
            NotImplementedError: If the dataset is not "conll2000".
        """
        if "conll2000" in data_path:
            nltk.download("conll2000")
            data = conll2000.tagged_sents(data_path.split("_")[-1])
            if self.tagging_tokenizer is None:
                self.tagging_tokenizer = POSTaggingTokenizer().fit(data)

            words_with_tag_ids = self.tagging_tokenizer(data)
            return words_with_tag_ids
        else:
            raise NotImplementedError(
                "Dataset not implemented for anything other than conll2000"
            )

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Words and corresponding tag IDs.
        """
        return [elem[0] for elem in self.data[idx]], [
            elem[1] for elem in self.data[idx]
        ]


class Collator:
    """
    Collator class for preparing batches of data.

    Args:
        tokenizer (transformers.AutoTokenizer): Tokenizer for encoding sentences into token IDs.
        post_tagging_tokenizer (POSTaggingTokenizer): Tokenizer for encoding POS tags into tag IDs.
        max_length (int, optional): Maximum length of tokenized sequences. Defaults to 512.
        padding (bool, optional): Whether to pad sequences. Defaults to True.
        truncation (bool, optional): Whether to truncate sequences. Defaults to True.
        add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.
        **tokenizer_kwargs: Additional arguments for the tokenizer.

    Attributes:
        tokenizer (transformers.AutoTokenizer): Tokenizer for encoding sentences.
        post_tagging_tokenizer (POSTaggingTokenizer): Tokenizer for encoding POS tags.
        tokenizer_kwargs (dict): Arguments for the tokenizer.
    """

    def __init__(
        self,
        tokenizer,
        post_tagging_tokenizer,
        max_length=512,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        **tokenizer_kwargs
    ):
        self.tokenizer = tokenizer
        self.post_tagging_tokenizer = post_tagging_tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        if self.tokenizer_kwargs.get("max_length", None) is None:
            self.tokenizer_kwargs["max_length"] = max_length
        if self.tokenizer_kwargs.get("padding", None) is None:
            self.tokenizer_kwargs["padding"] = padding
        if self.tokenizer_kwargs.get("truncation", None) is None:
            self.tokenizer_kwargs["truncation"] = truncation
        if self.tokenizer_kwargs.get("add_special_tokens", None) is None:
            self.tokenizer_kwargs["add_special_tokens"] = add_special_tokens

    def __call__(self, batch):
        """
        Prepares and tokenizes a batch of data.

        Args:
            batch (list): List of samples, where each sample is a tuple of words
                and their corresponding POS tags.

        Returns:
            tuple: Input IDs, attention masks, and POS tags per token.
        """
        word_lists = [elem[0] for elem in batch]
        tags_per_word = [elem[1] for elem in batch]

        sentences = [" ".join(word_list) for word_list in word_lists]

        tokenized_sentences = self.tokenizer(
            sentences, return_tensors="pt", **self.tokenizer_kwargs
        )
        input_ids = tokenized_sentences["input_ids"]
        attention_mask = tokenized_sentences["attention_mask"]

        tags_per_token = []

        for i in range(len(word_lists)):
            cur_sentence_tags_per_token = []
            cur_word_idx = 0
            tokens = [
                elem.strip() for elem in self.tokenizer.batch_decode(input_ids[i])
            ]
            for token in tokens:
                if token not in sentences[i]:
                    cur_sentence_tags_per_token.append(
                        self.post_tagging_tokenizer.pad_token_id
                    )
                else:
                    while not token in word_lists[i][cur_word_idx]:
                        cur_word_idx += 1
                    cur_sentence_tags_per_token.append(tags_per_word[i][cur_word_idx])

            tags_per_token.append(cur_sentence_tags_per_token)

        if self.tokenizer_kwargs["padding"]:
            tags_per_token = torch.tensor(tags_per_token, dtype=torch.long)

        return input_ids, attention_mask, tags_per_token
