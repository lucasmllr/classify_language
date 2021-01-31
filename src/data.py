import string
from unidecode import unidecode
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision.transforms.functional import pad


LANGUAGES = ['Estonian', 'Swedish', 'Dutch',
    'Turkish', 'Latin', 'Portugese', 'French', 'Spanish',
    'Romanian', 'English']


def prep_data(data, languages=LANGUAGES):
    data = data[data.language.isin(languages)]
    data.Text = data.Text.map(unidecode)
    translation = str.maketrans('', '', string.punctuation + string.digits)
    data.Text = data.Text.str.translate(translation)
    data.Text = data.Text.map(lambda s: ' '.join(s.split()))
    data.Text = data.Text.str.lower()
    data.Text = data.Text.replace('', np.nan)
    data = data.dropna()
    return data


def test_data(data):
    assert not any([data.Text.str.contains(c, regex=False).any() for c in string.punctuation]), 'data contains punctuation'
    assert not any([data.Text.str.contains(w).any() for w in string.whitespace[1:]]), 'data contains additional white spaces'
    assert data.Text.str.islower().all(), 'data is not all lower case'
    assert set(' '.join(data.Text.tolist())) == set(string.ascii_lowercase + ' '), 'data is not lower case ascii only'
    print('looking good')


class SimpleTextDataset(Dataset):

    def __init__(self, data, vocab, labels, input_dim=None):
        self.vocab = vocab
        if input_dim is not None:
            self.input_dim = input_dim
        self.input_dim = input_dim
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data.index)

    def tokenize(self, text):
        tokens = torch.tensor([self.vocab.index(char) for char in text]).long()
        return one_hot(tokens, len(self.vocab))

    def __getitem__(self, idx):
        # TODO: add augmenting start position
        text = self.data.iloc[idx].text
        text = self.tokenize(text)
        if hasattr(self, 'input_dim'):
            diff = len(text) - self.input_dim
            if diff >= 0:
                text = text[:self.input_dim]
            else:
                text = pad(text, [0, 0, 0, -diff])
        text = text.transpose(0, 1)
        label = self.data.iloc[idx].label
        label = torch.tensor(self.labels.index(label))
        return text.float(), label.long()
        

if __name__ == '__main__':

    src = '../data/dataset.csv'
    data = pd.read_csv(src)
    data = prep_data(data)
    test_data(data)
    data.to_csv('../data/prepped.csv')