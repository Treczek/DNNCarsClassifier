from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar

from cars.config import get_project_structure

from cars.training import StanfordCarsLightningModule
from cars.utils import configure_default_logging, Config

STRUCTURE = get_project_structure()
seed_everything(42)


class TrainingConsole:
    def __init__(self, config_path):

        self.log = configure_default_logging("cars")
        self.config = Config(config_path, logger=self.log)

        self.model = self._create_lightning_neural_module()
        self.trainer = self._create_lightning_trainer()

    def train_model(self):
        self.log.info("Training started.")
        self.trainer.fit(self.model)
        self.trainer.test(self.model)

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

        trainer = Trainer(
            deterministic=True,
            checkpoint_callback=checkpoint,
            callbacks=[progress_bar],
            **self.config['trainer']
        )

        return trainer


if __name__ == "__main__":
    console = TrainingConsole(r"config_template.yaml")
    console.train_model()
