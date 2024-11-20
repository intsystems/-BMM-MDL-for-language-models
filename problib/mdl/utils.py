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
    def __init__(self):
        """
        Constructor for POSTaggingTokenizer class that converts a list of words and their part-of-speech tags into a list of word-ids and tag-ids

        Notes
        -----
            Initializes the tag2idx and idx2tag dictionaries and sets the IDs for
            the special tags <PAD>, <BOS>, <EOS>, and <UNK>.
        """
        self.idx2tag = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.tag2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    def __len__(self):
        return len(self.idx2tag)
    
    def fit(self, sentences_conll):
        """
        Learns the tag vocabulary from a list of sentences in the CoNLL format

        Parameters
        ----------
        sentences_conll : List[List(tuple)] 
            A list of sentences where each sentence is a list of tuples (word, tag)

        Returns
        -------
        POSTaggingTokenizer (self)
            The same tokenizer with the learned tag vocabulary
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
        Converts a list of sentences in the CoNLL format to a list of sentences where
        each word is replaced with its tag ID. 

        Parameters
        ----------
        sent_words_with_tags : List[List(tuple)] 
            A list of sentences represented as a list of tuples (word, tag) (CoNLL format)

        Returns:
        List[List(tuple)] | List(tuple)
            A list of sentences where each word is replaced with its tag ID. If the input
            contains a single sentence, the output is a single sentence, otherwise it
            is a list of sentences.
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
        Converts a list of tag IDs to a list of part-of-speech tags

        Parameters
        ----------
        ids (list of int): A list of tag IDs

        Returns
        -------
        list
            A list of POS tags corresponding to the tag IDs
        """
        tags = []
        for idx in ids:
            if self.idx2tag.get(idx) is not None:
                tags.append(self.idx2tag[idx])
            else:
                tags.append(self.unk_token_id)
        return tags


class MDLDataset_POSTagging(Dataset):
    def __init__(
        self,
        data_path,
        tagging_tokenizer=None
    ):
        """
        Constructor for MDLDataset_POSTagging class

        Parameters
        ----------
        data_path : str
            either "conll2000_train.txt" or "conll2000_test.txt" otherwise raises NotImplementedError
        tagging_tokenizer : POSTaggingTokenizer, optional
            Part-of-speech tagging tokenizer to use for encoding the data. If None (default), a new tokenizer will be created and fit to the data.

        Returns
        -------
        MDLDataset_POSTagging
            A dataset containing the data in the given file, where each item is a list of words and part-of-speech tags
        """
        super().__init__()
        self.tagging_tokenizer = tagging_tokenizer
        self.data = self._read_data(data_path)

    def _read_data(self, data_path):
                
        """
        Reads data from a specified file path, processes it using the tagging_tokenizer that
        converts each sentence into a list of words with their corresponding tag IDs.

        Parameters
        ----------
        data_path : str
            The path to the data file. Should contain "conll2000" for the dataset to be processed.

        Returns
        -------
        list
            A list of words with their tag IDs, where each item is a tuple (word, tag_id).

        Raises
        ------
        NotImplementedError
            If the dataset specified in the data_path is not "conll2000".
        """
        if "conll2000" in data_path:
            nltk.download("conll2000")
            data = conll2000.tagged_sents(data_path.split("_")[-1])
            if self.tagging_tokenizer is None:
                self.tagging_tokenizer = POSTaggingTokenizer().fit(data)

            words_with_tag_ids = self.tagging_tokenizer(data)
            return words_with_tag_ids
        else:
            raise NotImplementedError("Dataset not implemented for anything other than conll2000")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve from the dataset.

        Returns
        -------
        tuple
            A tuple where the first element is a list of words and the second
            element is a list of corresponding POS tag IDs.
        """
        return [elem[0] for elem in self.data[idx]], [elem[1] for elem in self.data[idx]]


class Collator:
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
        """
        Initializes the Collator class for processing batches of data before dataloader.

        Parameters
        ----------
        tokenizer : transformers.AutoTokenizer
            Tokenizer used for encoding sentences into token IDs.
        post_tagging_tokenizer : POSTaggingTokenizer
            Tokenizer used for encoding part-of-speech tags into tag IDs.
        max_length : int, optional
            Maximum length of the tokenized sequences. Default is 512.
        padding : bool, optional
            Whether to pad the sequences to the max_length. Default is True.
        truncation : bool, optional
            Whether to truncate the sequences to the max_length. Default is True.
        add_special_tokens : bool, optional
            Whether to add special tokens to the sequences. Default is True.
        **tokenizer_kwargs : dict
            Additional keyword arguments passed to the tokenizer.
        """
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
        Collates and tokenizes a batch of data, alligns words part-of-speech tags with the corresponding tokens

        Parameters
        ----------
        batch : list
            List of tuples, where each tuple contains a list of words and a list of
            corresponding part-of-speech tags.

        Returns
        -------
        input_ids : torch.Tensor
            Tensor of token IDs, shape (batch_size, max_length).
        attention_mask : torch.Tensor
            Tensor of attention masks, shape (batch_size, max_length).
        tags_per_token : torch.Tensor
            Tensor of part-of-speech tags per token, shape (batch_size, max_length).
        """
        word_lists = [elem[0] for elem in batch]
        tags_per_word = [elem[1] for elem in batch]

        sentences = [" ".join(word_list) for word_list in word_lists]

        tokenized_sentences = self.tokenizer(sentences, return_tensors="pt", **self.tokenizer_kwargs)
        input_ids = tokenized_sentences["input_ids"]
        attention_mask = tokenized_sentences["attention_mask"]

        tags_per_token = []

        for i in range(len(word_lists)):
            cur_sentence_tags_per_token = []
            cur_word_idx = 0
            tokens = [elem.strip() for elem in self.tokenizer.batch_decode(input_ids[i])]
            for token in tokens:
                if token not in sentences[i]:
                    cur_sentence_tags_per_token.append(self.post_tagging_tokenizer.pad_token_id)
                else:
                    while not token in word_lists[i][cur_word_idx]:
                        cur_word_idx += 1
                    cur_sentence_tags_per_token.append(tags_per_word[i][cur_word_idx])
            
            tags_per_token.append(cur_sentence_tags_per_token)

        if self.tokenizer_kwargs["padding"]:
            tags_per_token = torch.tensor(tags_per_token, dtype=torch.long)

        return input_ids, attention_mask, tags_per_token
