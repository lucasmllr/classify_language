import torch
import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from os.path import join


class LanguageClassifier(pl.LightningModule):

    def __init__(self, params, model, labels):
        super(LanguageClassifier, self).__init__()
        self.params = params
        self.model = model
        self.labels = labels

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
        inpt = torch.argmax(inpt, dim=1).long()
        with open(join(self.params.saving.root, f'{mode}_result_{self.current_epoch}.csv'), 'w+') as f:
            f.write('target, prediction, input\n')
            for b in range(inpt.shape[0]):
                line = ''.join([self.params.model.vocab[i] for i in list(inpt[b])])
                t = self.labels[target[b]]
                p = self.labels[pred_labels[b]]
                line = f'{t}, {p}, ' + line + '\n'
                f.write(line)
