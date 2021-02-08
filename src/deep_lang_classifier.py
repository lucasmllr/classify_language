import torch
import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from os.path import join


class LanguageClassifier(pl.LightningModule):
    '''defines the training procedure for a deep learning based language classification model.
    It wraps a PyTorch Module and defines that takes as input a text sequence and computes 
    a classification output. The model can have character or word based vocabulary. 
    '''

    def __init__(self, params, model, labels, vocab):
        '''
        Args:
            params (DotMap): parameter object
            model (Module): pytorch classification model
            labels (list): of labels, order must agree with dataloading
            vocab (dict): of the form {item : index}, indices must order items alphabetically
        '''
        super(LanguageClassifier, self).__init__()
        self.params = params
        self.model = model
        self.labels = labels
        self.vocab = vocab
        self.words = sorted(list(self.vocab.keys())) + ['.']

    def forward(self, inpt):
        return self.model(inpt)

    def _step(self, inpt, target):
        pred = self.forward(inpt)
        loss = cross_entropy(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        acc = torch.sum(pred_labels == target) / len(target)
        return loss, acc, pred_labels

    def training_step(self, batch, batch_idx):
        inpt, target = batch
        loss, acc, pred_labels = self._step(inpt, target)
        self.log_dict({'train_loss':loss, 'train_acc':acc}, prog_bar=True)
        if self.current_epoch % self.params.saving.period == 0 and batch_idx == 0:
            self._save_intermediate_results(inpt, pred_labels, target, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        inpt, target = batch
        loss, acc, pred_labels = self._step(inpt, target)
        self.log_dict({'val_loss':loss, 'val_acc':acc}, prog_bar=True)
        if self.current_epoch % self.params.saving.period == 0 and batch_idx == 0:
            self._save_intermediate_results(inpt, pred_labels, target, mode='val')
        return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.params.training.lr)

    def _save_intermediate_results(self, inpt, pred_labels, target, mode):
        if self.params.model.vocab_type == 'char':
            inpt = torch.argmax(inpt, dim=1).long()
        with open(join(self.params.saving.root, self.params.saving.name, f'{mode}_result_{self.current_epoch}.csv'), 'w+') as f:
            f.write('target, prediction, input\n')
            for b in range(inpt.shape[0]):
                if self.params.model.vocab_type == 'char':
                    line = ''.join([self.words[c] for c in list(inpt[b])])
                elif self.params.model.vocab_type == 'word':
                    line = ' '.join([self.words[w] for w in list(inpt[b])])
                else:
                    raise ValueError('vocab_type needs to be "char" or "word".')
                t = self.labels[target[b]]
                p = self.labels[pred_labels[b]]
                line = f'{t}, {p}, ' + line + '\n'
                f.write(line)

    def predict_df(self, df):
        self.eval()
        data = df.copy()
        data.tokens = data.tokens.map(lambda x: x.unsqueeze(0))
        data['pred'] = data.tokens.map(self.forward)
        data.pred = data.pred.map(lambda x: torch.argmax(x.squeeze()))
        data.pred = data.pred.map(lambda x: self.labels[x])
        return data

    def load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt['state_dict']
        self.load_state_dict(state_dict)
