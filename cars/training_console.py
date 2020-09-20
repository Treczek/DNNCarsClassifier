import os
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
from pytorch_lightning.loggers import NeptuneLogger

from cars.config import get_project_structure
from cars.training import StanfordCarsLightningModule
from cars.utils import configure_default_logging, Config

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

        progress_bar = ProgressBar(refresh_rate=1)
        neptune_logger = self._initialize_neptune_connection()

        trainer = Trainer(
            checkpoint_callback=checkpoint,
            callbacks=[progress_bar],
            logger=neptune_logger,
            **self.config['trainer']
        )

        return trainer

    def _initialize_neptune_connection(self):
        if self.config['neptune:enabled']:
            neptune_parameters = {
                'architecture': self.lightning_module.model.__class__.__name__,
                'num_params': sum(p.numel() for p in self.lightning_module.model.parameters() if p.requires_grad),
                'img_size': self.config['preprocessing:image_size'],
                # 'grayscale': CFG['convert_to_grayscale'],
                # 'normalize': CFG['normalize'],
                # 'norm_params_rgb': CFG['normalization_params_rgb'] if CFG['normalize'] and not CFG[
                #     'convert_to_grayscale'] else None,
                # 'norm_params_gray': CFG['normalization_params_grayscale'] if CFG['normalize'] and CFG[
                #     'convert_to_grayscale'] else None,
                # 'crop_to_bboxes': CFG['crop_to_bboxes'],
                # 'erase_background': CFG['erase_background'],
                # 'augment_images': CFG['augment_images'],
                # 'image_augmentations': CFG['image_augmentations'] if CFG['augment_images'] else None,
                # 'augment_tensors': CFG['augment_tensors'],
                # 'tensor_augmentations': CFG['tensor_augmentations'] if CFG['augment_tensors'] else None,
                'batch_size': self.config['experiment:batch_size'],
                'max_num_epochs': self.config['trainer:max_epochs'],
                # 'dropout': CFG['dropout'],
                # 'out_channels': CFG['out_channels'],
                'loss_function': self.config["experiment:loss_function"].__class__.__name__,
                # 'loss_params': CFG['loss_params'],
                'optimizer': self.config["experiment:optimizer"].__name__,
                'learning_rate': self.config["experiment:learning_rate"],
                # 'weight_decay': OPTIMIZER_PARAMS['weight_decay'] if OPTIMIZER_PARAMS.get(
                #     'weight_decay') is not None else 0.0,
                # 'all_optimizer_params': OPTIMIZER_PARAMS,
                # 'lr_scheduler': LR_SCHEDULER.__name__ if LR_SCHEDULER is not None else None,
                # 'lr_scheduler_params': LR_SCHEDULER_PARAMS
            }

            neptune_logger = NeptuneLogger(
                api_key=os.environ['neptune_api_token'],
                project_name='treczek/stanford-cars',
                experiment_name=self.config['model:name'].__class__.__name__ + '_' + datetime.now().strftime('%Y%m%d%H%M%S'),
                params=neptune_parameters
            )
        else:
            neptune_logger = None

        return neptune_logger


if __name__ == "__main__":
    console = TrainingConsole(r"config_template.yaml")
    console.train_model()
