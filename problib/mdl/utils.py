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
        self.idx2tag = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.tag2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    def __len__(self):
        return len(self.idx2tag)
    
    def fit(self, sentences_conll):
        idx = len(self.idx2tag)
        for s in sentences_conll:
            for _, word_tag in s:
                if self.tag2idx.get(word_tag) is None:
                    self.tag2idx[word_tag] = idx
                    self.idx2tag[idx] = word_tag
                    idx += 1
        return self

    def encode(self, sent_words_with_tags):
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
        return self.encode(NERs)

    def decode(self, ids):
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
        super().__init__()
        self.tagging_tokenizer = tagging_tokenizer
        self.data = self._read_data(data_path)

    def _read_data(self, data_path):
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
