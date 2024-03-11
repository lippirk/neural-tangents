import torch as t
from torch.utils.data import Dataset
import string
import random
from typing import Union, Tuple

import contextlib

@contextlib.contextmanager
def random_ctx(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)

class RandomStringDataset(Dataset):
    def __init__(self, max_input_len=32, num_letters=26, dataset_size = 1000, special_tokens=[], fixed_len: Union[int, None]=None, seeded: bool=False):
        """
        generates random inputs for a transformer model. supports random length inputs. handles padding, but not masking!
        Args:
          max_input_len: maximum length of generated input
          fixed_len: if provided, fixes the length of the randomly generated inputs
          num_letters: number of letters in the alphabet
          dataset_size: we need to implement [__len__], so we need to know the size of the dataset
          special_tokens: list of special tokens to add to the vocabulary, e.g. <copy>
        """
        self.max_input_len = max_input_len
        self.num_letters = num_letters
        self.chars = [x for x in string.ascii_lowercase[:num_letters]]
        self.dataset_size = dataset_size
        our_special_tokens = ['<bos>', '<eos>', '<pad>', '<mask>']
        assert len(set(special_tokens).intersection(set(our_special_tokens))) == 0
        assert len(set(special_tokens).intersection(set(self.chars))) == 0
        self.all_tokens = our_special_tokens + special_tokens + self.chars
        self.ctoi = {x: i for i, x in enumerate(self.all_tokens)}
        self.itoc = {i: x for i, x in enumerate(self.all_tokens)}
        self.fixed_len = fixed_len
        self.seeded = seeded
        self.padding_idx = self.ctoi['<pad>']
    def decode(self, xs: list): return [self.itoc[x] for x in xs]
    def __len__(self):
        return self.dataset_size
    def get_x_y(self, input_str) -> Tuple[list, list, list]:
        """
        Args:
          input_str: a randomly generated list of chars
        Returns:
            tuple x,y,mask. expected them to be lists of equal length
            x: list of tokens that will be fed to model
            y: list of tokens that we expect model to predict
            mask: list of 0s and 1s, 0s indicated tokens to ignore
        """
        raise NotImplementedError("Subclasses must implement this method")
    def __getitem__(self, idx):
        x, y, tgt_mask, pad_mask  = self.get_sample_raw(idx)


        # tensorize
        x = t.tensor([self.ctoi[xi] for xi in x])
        y = t.tensor([self.ctoi[yi] for yi in y])
        tgt_mask = t.tensor(tgt_mask)
        pad_mask = t.tensor(pad_mask)
        return x, y, tgt_mask, pad_mask
    def get_sample_raw(self, idx):
        input_str = self.random_string(idx)
        x,y,tgt_mask = self.get_x_y(input_str)
        assert len(x) == len(y) == len(tgt_mask)
        # padding
        if len(x) < self.ctx_size:
            padding_mask = [1]*len(x) + [0]*(self.ctx_size - len(x))
            x = x + ['<pad>'] * (self.ctx_size - len(x))
            y = y + ['<pad>'] * (self.ctx_size - len(y))
            tgt_mask = tgt_mask + [0] * (self.ctx_size - len(tgt_mask))
        else:
            padding_mask = [1]*self.ctx_size
        assert len(x) == len(y) == len(tgt_mask) == len(padding_mask) == self.ctx_size

        return x,y,tgt_mask,padding_mask
    def unseeded_random_string(self):
        if self.fixed_len is not None:
            length = self.fixed_len
        else:
            length = random.randint(self.max_input_len//2, self.max_input_len)
        return random.choices(self.chars, k=length)
    def random_string(self, idx):
        if self.seeded:
            with random_ctx(idx):
                return self.unseeded_random_string()
        else:
            return self.unseeded_random_string()

class SelectTokenTaskDataset(RandomStringDataset):
    """objective is to repeat the first token presented (that isn't the bos token)
       [k] times"""
    def __init__(self, k=3, which_token='first', **kwargs):
        super().__init__(special_tokens=['<fst>'], **kwargs)
        self.ctx_size = kwargs['max_input_len'] + 2 + k
        self.k = k
        self.which_token = which_token
    def get_x_y(self, input_str):
        exp_out = input_str[0] if self.which_token == 'first' else input_str[-1]
        x = ['<bos>'] + input_str + ['<fst>'] + [exp_out]*self.k
        y = ['<mask>'] * (len(input_str) + 1) + [exp_out]*self.k + ['<eos>']
        tgt_mask = [0] * (len(input_str)+1)  + [1] * (len(x) - 1 - len(input_str))
        return x, y, tgt_mask

class CopyTaskDataset(RandomStringDataset):
    """See 'the copy task' from Section 2 in https://arxiv.org/pdf/2402.01032.pdf"""
    def __init__(self, **kwargs):
        super().__init__(special_tokens=['<copy>'], **kwargs)
        self.ctx_size = 2*kwargs['max_input_len'] + 2
    def get_x_y(self, input_str):
        x = ['<bos>'] + input_str + ['<copy>'] + input_str
        y = ['<mask>'] * (len(input_str) + 1) + input_str + ['<eos>']
        tgt_mask = [0] * (len(input_str)+1)  + [1] * (len(input_str) + 1)
        return x, y, tgt_mask
class SuffixKeyLookupTaskDataset(RandomStringDataset):
    """See n-gram lookup task from Section 3 in https://arxiv.org/pdf/2402.01032.pdf"""
    def __init__(self, k=3, ngram_length=3, **kwargs):
        super().__init__(special_tokens=['<copy>'], **kwargs)
        self.k = k
        self.ctx_size = 2*kwargs['max_input_len'] + 2
        self.ngram_length = ngram_length
    def get_x_y(self, input_str):
        # randomly select a start idx
        start_ix_max = len(input_str) - self.ngram_length - self.k - 1
        if 0 >= start_ix_max:
            raise ValueError(f"make [max_input_len] longer, it is currently too small: {self.max_input_len}")
        start_ix = random.randint(0, start_ix_max)
        end_ix = start_ix + self.ngram_length
        assert end_ix + self.k < len(input_str), f"make [max_input_len] longer, currently too small: {self.max_input_len}"
        prompt = input_str[start_ix:end_ix]
        x = ['<bos>'] + input_str + ['<copy>'] + prompt
        y = ['<mask>'] * (len(input_str)+1) + input_str[end_ix:(end_ix + self.k)] + ['<eos>']
        tgt_mask = [0] * (len(input_str)+1)  + [1] * (len(x) - 1 - len(input_str))
        return x, y, tgt_mask

if __name__ == '__main__':
    print("first token task samples:\n====")
    for i in range(3):
        print(SelectTokenTaskDataset(max_input_len=4).get_sample_raw(i))
    print("copy task samples:\n====")
    for i in range(3):
        print(CopyTaskDataset(max_input_len=22, fixed_len=11).get_sample_raw(i))
        print(CopyTaskDataset(max_input_len=22).get_sample_raw(i))
    print("suffix key lookup taks samples:\n====")
    for i in range(3):
        print(SuffixKeyLookupTaskDataset(max_input_len=10, fixed_len=10).get_sample_raw(i))
        print(SuffixKeyLookupTaskDataset(max_input_len=24).get_sample_raw(i))