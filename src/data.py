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
    '''preprocesses a data frame containing lines of text in different languages.
    Defined languages are selected, all characters are transformed to ascii lowercase,
    punctiation, digits and additional white spaces are removed.

    Args:
        data (DataFrame): must contain columns "Text" and "language"
        languages (list): of strings defining selected languages in data.language
    '''
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
    '''tests whether the contained data does not contain punctuation, no white spaces except " ", 
    is lowercase and only contains ascii characters.
    '''
    assert not any([data.Text.str.contains(c, regex=False).any() for c in string.punctuation]), 'data contains punctuation'
    assert not any([data.Text.str.contains(w).any() for w in string.whitespace[1:]]), 'data contains additional white spaces'
    assert data.Text.str.islower().all(), 'data is not all lower case'
    assert set(' '.join(data.Text.tolist())) == set(string.ascii_lowercase + ' '), 'data is not lower case ascii only'
    print('looking good')


class CharTextDataset(Dataset):
    '''implements a PyTorch Dataset for character level text classification with a fixed sequence length.
    Lines of text plus their labels are stored in a pandas DataFrame. 
    When an instance is accessed characters are one-hot-encoded according to a defined vocabulary and 
    labels are transformed to integers.
    '''

    def __init__(self, data, vocab, labels, input_dim):
        '''
        Args:
            data (DataFrame): must contain columns "text" and "label"
            vocab (dict): vocaulary dictionart of the form {character : index}
            labels (list): list of labels appearing in data.label
            input_dim (int): length of returned sequence
        '''
        self.vocab = vocab
        self.input_dim = input_dim
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data.index)

    def tokenize(self, text):
        tokens = torch.tensor([self.vocab[char] for char in text]).long()
        tokens = one_hot(tokens, len(self.vocab))
        diff = len(tokens) - self.input_dim
        if diff >= 0:
            tokens = tokens[:self.input_dim]
        else:
            tokens = pad(tokens, [0, 0, 0, -diff])
        return tokens.transpose(0, 1).float()

    def __getitem__(self, idx):
        # TODO: add augmenting start position
        text = self.data.iloc[idx].text
        text = self.tokenize(text)
        label = self.data.iloc[idx].label
        label = torch.tensor(self.labels.index(label))
        return text, label.long()
        
    def get_tokenized_data(self):
        data = self.data.copy()
        data['tokens'] = data.text.map(self.tokenize)
        return data


class WordTextDataset(Dataset):
    '''implements a PyTorch Dataset for word level text classification with a fixed sequence length.
    Lines of text plus their labels are stored in a pandas DataFrame. 
    When an instance is accessed words are transformed to indices according to a defined vocabulary and 
    labels are transformed to integers.
    '''
    # TODO: there should be an abstract text dataset class
    # TODO: only fixed length outputs so far
    def __init__(self, data, vocab, labels, input_dim):
        """
        Args:
            data (DataFrame): must contain columns "text" and "label"
            vocab (dict): with word, index pairs, word indices should be alphabetically ordered
            labels (list): list of labels appearing in data.label
            input_dim (int): length of input sequences
        """
        self.vocab = vocab
        self.data = data
        self.labels = labels
        self.input_dim = input_dim
        
    def __len__(self):
        return len(self.data.index)

    def tokenize(self, text):
        text = text.split(' ')
        tokens = [torch.tensor(self.vocab[w]).long() for w in text]
        tokens = torch.stack(tokens)
        diff = len(tokens) - self.input_dim
        if diff >= 0:
            tokens = tokens[:self.input_dim]
        else:
            padding = torch.full((-diff,), len(self.vocab))
            tokens = torch.cat([tokens, padding])
        return tokens

    def __getitem__(self, idx):
        # TODO: add augmenting start position
        text = self.data.iloc[idx].text
        text = self.tokenize(text)
        label = self.data.iloc[idx].label
        label = torch.tensor(self.labels.index(label)).long()
        return text, label

    def get_tokenized_data(self):
        data = self.data.copy()
        data['tokens'] = data.text.map(self.tokenize)
        return data



if __name__ == '__main__':

    src = '../data/dataset.csv'
    data = pd.read_csv(src)
    data = prep_data(data)
    test_data(data)
    data.to_csv('../data/prepped.csv')