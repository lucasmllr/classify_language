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
        if np.sum(hist) < 0.9:  # arbitrary threshold < 1
            return np.nan
        pred = np.argmax(hist, axis=0)
        return sorted(self.labels.keys())[pred]

    def predict_df(self, df):
        df['pred'] = df.text.map(self.predict)
        return df