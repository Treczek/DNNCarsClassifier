import os
from datetime import datetime

from pytorch_lightning import Trainer, Callback, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import NeptuneLogger

from cars.config import get_project_structure
from cars.training import StanfordCarsLightningModule
from cars.utils import configure_default_logging, Config, calculate_model_stats

STRUCTURE = get_project_structure()
seed_everything(42)


class TrainingConsole:
    def __init__(self, config_path):

        self.log = configure_default_logging("cars")
        self.config = Config(config_path, logger=self.log)

        self.lightning_module = self._create_lightning_neural_module()
        self.trainer = self._create_lightning_trainer()

    def train_model(self):
        self.log.info("Training started.")
        self.trainer.fit(self.lightning_module)
        self.trainer.test(self.lightning_module)

    def _create_lightning_neural_module(self):

        # Create neural architecture specified in config
        architecture = self.config["model:name"](**self.config["model:kwargs"])

        # Initialization of the lightning wrapper for created architecture
        lightning_module = StanfordCarsLightningModule(
            model=architecture,
            config=self.config,
            logger=self.log
        )

        return lightning_module

    def _create_lightning_trainer(self):
        checkpoint = ModelCheckpoint(
            filepath=STRUCTURE["output_data"],
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

        early_stop_callback = EarlyStopping(
            min_delta=0.0,
            patience=self.config["experiment:early_stop"],
            verbose=True,
            mode='min'
        )

        lr_monitor = LearningRateLogger()
        logging_callback = LoggingCallback()

        neptune_logger = self._initialize_neptune_connection()

        trainer = Trainer(
            checkpoint_callback=checkpoint,
            early_stop_callback=early_stop_callback,
            callbacks=[lr_monitor, logging_callback],
            logger=neptune_logger,
            gpus=1,
            num_sanity_val_steps=0,
            **self.config['trainer']
        )

        return trainer

    def _initialize_neptune_connection(self):

        used_augmentations = [augmentation for augmentation, is_used in self.config["preprocessing:augmentations"].items() if is_used]

        if self.config['neptune:enabled']:
            neptune_parameters = {
                'architecture': self.lightning_module.model.__class__.__name__,
                'scaling_parameter': self.config["model:kwargs:scaling_parameter"],
                'img_size': self.config['preprocessing:image_size'],
                'batch_size': self.config['experiment:batch_size'],
                'max_num_epochs': self.config['trainer:max_epochs'],
                'augmentation_used': used_augmentations,
                'augmentation_kwargs': {aug: self.config["preprocessing:augmentation_kwargs"][aug] for aug in used_augmentations},
                # 'dropout_p': self.config['dropout_p'],
                'loss_function': self.config["experiment:loss_function"].__class__.__name__,
                'optimizer': self.config["experiment:optimizer"].__name__,
                'learning_rate': self.config["experiment:optimizer_kwargs:lr"],
                'weight_decay': self.config['experiment:optimizer_kwargs:weight_decay'],
                'lr_scheduler': self.config["experiment:scheduler"].__name__,
                'lr_scheduler_kwargs': self.config['experiment:scheduler_kwargs']
            }

            neptune_logger = NeptuneLogger(
                api_key=os.environ['neptune_api_token'],
                project_name='treczek/stanford-cars',
                experiment_name=self.config['model:name'].__class__.__name__ + '_' + datetime.now().strftime(
                                                                                                        '%Y%m%d%H%M%S'),
                params=neptune_parameters
            )
        else:
            neptune_logger = None

        return neptune_logger


class LoggingCallback(Callback):
    def __init__(self):
        self.log = configure_default_logging("cars")

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        if trainer.logger is not None:
            model_params, model_flops = calculate_model_stats(trainer.model, image_size=trainer.model.image_size)
            trainer.logger.experiment.log_metric("num_params", model_params)
            trainer.logger.experiment.log_metric("model_flops", model_flops)
