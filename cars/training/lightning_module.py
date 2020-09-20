"""
This module will contain pytorch lightning object that will handle model training.
"""

import pytorch_lightning as pl
import torch

from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader
from torchvision import transforms

from cars.datasets import StanfordCarsDataset


class StanfordCarsLightningModule(pl.LightningModule):
    def __init__(self, model, config, logger):
        super().__init__()

        self.config = config
        self.log = logger

        self.model = model
        self.batch_size = self.config["experiment:batch_size"]
        self.image_size = self.config["preprocessing:image_size"]

        self.data_train = None
        self.data_test = None

        self.loss = self.config["experiment:loss_function"]

    @pl.core.decorators.auto_move_data
    def forward(self, input):
        return self.model.forward(input)

    def prepare_data(self):

        transform_test = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            # transforms.Grayscale(),
            transforms.ToTensor()
        ])

        transform_train = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            # transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(**self.config["preprocessing:random_affine"]),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
        ])

        self.data_train = StanfordCarsDataset(
            mode="train",
            image_size=self.image_size,
            transformer=transform_train,
            logger=self.log
        )

        self.log.info("Train dataset created.")

        self.data_test = StanfordCarsDataset(
            mode="test",
            image_size=self.image_size,
            transformer=transform_test,
            logger=self.log
        )

        self.log.info("Test dataset created.")

    def step(self, batch, batch_idx, loss_type):
        input, labels = batch

        predictions = self.forward(input)
        pred_classes = torch.argmax(predictions, dim=1)

        loss = self.loss(predictions, labels)
        logs = dict(loss_type=loss,
                    accuracy=accuracy(pred_classes, labels))

        return {loss_type: loss, 'log': logs}

    def training_step(self, batch_train, batch_idx):
        return self.step(batch_train, batch_idx, 'loss')

    def test_step(self, batch_test, batch_idx):
        return self.step(batch_test, batch_idx, 'test_loss')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def configure_optimizers(self):
        # TODO Scheduler Plateau

        optimizer = self.config["experiment:optimizer"]
        self.log.info(f"Optimizer picked: {optimizer.__name__}")

        starting_lr = self.config["experiment:learning_rate"]
        self.log.info(f"Starting learning rate: {starting_lr}")

        return optimizer(self.parameters(), lr=starting_lr)
