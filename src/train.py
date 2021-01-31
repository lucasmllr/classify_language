import argparse
import string
from pandas import read_csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from os.path import join

from .utils import init_experiment_from_config
from .models import CharCNN
from .language_classifier import LanguageClassifier
from .data import SimpleTextDataset, test_data


# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', default='configs/config.yml', 
                    help='config file containing training params')
args = parser.parse_args()

params = init_experiment_from_config(args.config)

# data
print('loading data')
data = read_csv(params.data.path)
test_data(data)
data = data.rename(columns={params.data.text_column : 'text', params.data.label_column : 'label'})
labels = sorted(list(data.label.unique()))
train_data, val_data = train_test_split(data, test_size=params.data.val_split)
train_ds = SimpleTextDataset(
    data=train_data,
    vocab=params.model.vocab,
    labels=labels,
    input_dim=params.model.input_len
)
val_ds = SimpleTextDataset(
    data=val_data,
    vocab=params.model.vocab,
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
model = CharCNN(
    vocab_len=len(params.model.vocab),
    conv_features=params.model.conv_features, 
    fc_in_features=params.model.fc_in_features, 
    fc_features=params.model.fc_features, 
    n_classes=params.data.n_classes
)
langcla = LanguageClassifier(params, model, labels)

# training
print('initializing training')
logger = TensorBoardLogger(params.saving.root, name=params.saving.name)
checkpoint_callback = ModelCheckpoint(monitor='val_loss', verbose=True, save_last=True)
trainer = Trainer(
    gpus=None, 
    max_epochs=params.training.n_epochs,
    default_root_dir=join(params.saving.root, params.saving.name),
    logger=logger,
    checkpoint_callback=checkpoint_callback
)
trainer.fit(langcla, train_loader, val_loader)