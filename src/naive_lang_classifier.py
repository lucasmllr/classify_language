import numpy as np
from tqdm import tqdm


class NaiveLanguageClassifier:

    def __init__(self):
        self.words = {}
        self.labels = None

    def fit(self, data, max_idx=None):
        max_idx = max_idx or len(data.index)
        self.labels = {l : i for i, l in enumerate(sorted(data.label.unique()))}
        for i in tqdm(range(max_idx)):
            for w in data.iloc[i].text.split():
                l = self.labels[data.iloc[i].label]
                if not w in self.words.keys():
                    self.words[w] = [0] * len(self.labels)
                self.words[w][l] += 1
        for k, v in self.words.items():
            hist = np.array(v)
            self.words[k] = hist / np.sum(hist)

    def predict_hist(self, text):
        hists = []
        for w in text.split():
            if w in self.words.keys():
                hists.append(self.words[w])
        if not hists:  # no known word
            return np.zeros(len(self.labels))
        else:  # return histogram over labels
            hist = np.stack(hists)
            hist = np.sum(hist, axis=0) / np.sum(hist)
            return hist

    def predict(self, text):
        hist = self.predict_hist(text)
        if np.sum(hist) < 0.9:  # no known word
            return 'Dunno'
        pred = np.argmax(hist, axis=0)
        return sorted(self.labels.keys())[pred]

    def predict_df(self, df):
        df['pred'] = df.text.map(self.predict)
        return df


if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    import argparse
    from pandas import read_csv
    from os.path import join

    from .data import test_data
    from .utils import init_experiment_from_config
    from . import evaluation as eval

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/naive_config.yml', 
                        help='config file containing training params')
    args = parser.parse_args()

    params = init_experiment_from_config(args.config)

    print('loading data')
    data = read_csv(params.data.path)
    test_data(data)
    data = data.rename(columns={params.data.text_column : 'text', params.data.label_column : 'label'})
    train_data, val_data = train_test_split(data, test_size=params.data.val_split)
    # train_data = train_data[:100]
    labels = sorted(list(data.label.unique()))

    print('fitting naive classifier')
    langcla = NaiveLanguageClassifier()
    langcla.fit(train_data)

    print('evaluating')
    pred_df = langcla.predict_df(val_data)
    print('saving results')
    eval.confusion(
        pred_df, labels, 
        dst=join(params.saving.root, params.saving.name, 'confusion.pdf'),
        title=params.saving.name + ' Confusion'
    )

    eval.failures(
        pred_df, 
        dst=join(params.saving.root, params.saving.name, 'fails.csv'),
        n_max=20
    )
