
import argparse
from src.models.bow import BOWClassifier
import string
from dotmap import DotMap
from pandas import read_csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from os.path import join

from .utils import init_experiment_from_config
from .models.cnn import CharCNN
from .deep_lang_classifier import LanguageClassifier
from .data import WordTextDataset, WordTextDataset, test_data


def train_with(params:DotMap):
    # data
    print('loading data')
    data = read_csv(params.data.path)
    test_data(data)
    data = data.rename(columns={params.data.text_column : 'text', params.data.label_column : 'label'})
    labels = sorted(list(data.label.unique()))
    # character based vocabulary
    # vocab = {c : i for i, c in enumerate(sorted(list(' ' + string.ascii_lowercase)))}
    # word based vocabulary
    text = ' '.join(data.text.to_list()).split()
    vocab = {w : i for i, w in enumerate(sorted(set(text)))}
    train_data, val_data = train_test_split(data, test_size=params.data.val_split)
    train_ds = WordTextDataset(
        data=train_data,
        vocab=vocab,
        labels=labels,
        input_dim=params.model.input_len
    )
    val_ds = WordTextDataset(
        data=val_data,
        vocab=vocab,
        labels=labels,
        input_dim=params.model.input_len
    )
    train_loader = DataLoader(
        train_ds, 
        shuffle=True, 
        batch_size=params.training.batch_size, 
        num_workers=params.training.n_workers
    )
    val_loader = DataLoader(
        val_ds, 
        shuffle=True, 
        batch_size=params.training.batch_size, 
        num_workers=params.training.n_workers
    )

    # model
    print('initializing model')
    # model = CharCNN(
    #     vocab_len=len(vocab),
    #     conv_features=params.model.conv_features, 
    #     fc_in_features=params.model.fc_in_features, 
    #     fc_features=params.model.fc_features, 
    #     n_classes=params.data.n_classes
    # )
    model = BOWClassifier(
        vocab_len=len(vocab),
        hidden_dim=params.model.hidden_dim,
        n_classes=params.data.n_classes
    )
    langcla = LanguageClassifier(params, model, labels, vocab)

    # training
    print('initializing training')
    logger = TensorBoardLogger(params.saving.root, name=params.saving.name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', verbose=True, save_last=True)
    trainer = Trainer(
        gpus=params.training.gpus, 
        max_epochs=params.training.n_epochs,
        default_root_dir=join(params.saving.root, params.saving.name),
        logger=logger,
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(langcla, train_loader, val_loader)


if __name__ == '__main__':

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/config.yml', 
                        help='config file containing training params')
    args = parser.parse_args()

    params = init_experiment_from_config(args.config)

    train_with(params)