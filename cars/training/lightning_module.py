"""
This module will contain pytorch lightning object that will handle model training.
"""

import pytorch_lightning as pl
import torch

from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import (Grayscale,
                                    ColorJitter,
                                    RandomAffine,
                                    RandomErasing,
                                    RandomHorizontalFlip,
                                    RandomPerspective,
                                    ToTensor,
                                    RandomRotation,
                                    Resize,
                                    Compose)

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

    def forward(self, input):
        return self.model.forward(input)

    def prepare_data(self):

        augmentation_dict = dict(
            grayscale=Grayscale,
            random_affine=RandomAffine,
            color_jitter=ColorJitter,
            random_erasing=RandomErasing,
            horizontal_flip=RandomHorizontalFlip,
            random_perspective=RandomPerspective,
            random_rotation=RandomRotation
        )

        train_transformations = [Resize((self.image_size, self.image_size))]
        for augmentation, is_used in self.config["preprocessing:augmentations"]:
            if is_used:
                train_transformations.append(
                  augmentation_dict[augmentation](**self.config[f"preprocessing:augmentations_kwargs:{augmentation}"]))
        train_transformations.append(ToTensor())

        test_transformations = [Resize((self.image_size, self.image_size))]
        if self.config["preprocessing:augmentations:grayscale"]:
            test_transformations.append(Grayscale())
        test_transformations.append(ToTensor())

        transform_train = Compose(train_transformations)
        transform_test = Compose(test_transformations)

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

    def training_step(self, batch_train, batch_idx):
        input, labels = batch_train
        preds = self.forward(input)
        pred_classes = torch.argmax(preds, dim=1)

        loss = self.loss(preds, labels)
        acc = accuracy(pred_classes, labels, num_classes=self.model.num_classes)

        result = pl.TrainResult(loss)
        result.log_dict({
            'train_loss': loss,
            'train_acc': acc
        }, on_step=False, on_epoch=True)
        return result

    def validation_step(self, batch_test, batch_idx):
        input, labels = batch_test
        preds = self.forward(input)
        pred_classes = torch.argmax(preds, dim=1)

        loss = self.loss(preds, labels)
        acc = accuracy(pred_classes, labels, num_classes=self.model.num_classes)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({
            'valid_loss': loss,
            'valid_acc': acc
        }, on_step=False, on_epoch=True)

        return result

    def test_step(self, batch_test, batch_idx):
        return self.validation_step(batch_test, batch_idx)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def configure_optimizers(self):

        optimizer = self.config["experiment:optimizer"](self.parameters(), **self.config["experiment:optimizer_kwargs"])
        self.log.info(f"Optimizer picked: {optimizer.__class__.__name__}")

        starting_lr = self.config["experiment:optimizer_kwargs:lr"]
        self.log.info(f"Starting learning rate: {starting_lr}")

        try:
            scheduler = self.config["experiment:scheduler"]
            self.log.info(f"Scheduler picked: {scheduler.__name__}")
            scheduler = dict(
                scheduler=scheduler(optimizer, **self.config["experiment:scheduler_kwargs"]),
                monitor='val_checkpoint_on',
                name='lr'
            )
        except KeyError:
            return [optimizer]

        return [optimizer], [scheduler]
