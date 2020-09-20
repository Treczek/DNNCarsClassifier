"""
Config class will be used across all modules. It will help to organize arguments inside the yaml file and get them
using given convention: section:subsection:argument
"""

import yaml
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adam, AdamW, SGD

from cars.models.mobile_nets import MobileNetV1, MobileNetV2, SmallMobileNetV3, LargeMobileNetV3


class Config:
    def __init__(self, config_path_or_dict, logger):

        self.log = logger

        if isinstance(config_path_or_dict, str):
            self._config = self._read_yaml(config_path_or_dict)
        elif isinstance(config_path_or_dict, dict):
            self._config = config_path_or_dict
        else:
            raise AttributeError("Config must be a dict or path")

        self._log_config_in_logger()
        self._update_trainer_kwargs()
        self._update_config_values()

    @staticmethod
    def _read_yaml(config_path):
        with open(config_path, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def __getitem__(self, name):
        nested_arguments = name.split(":")
        for i, argument in enumerate(nested_arguments):
            if i == 0:
                return_value = self._config[argument]
            else:
                return_value = return_value[argument]
        return return_value

    def __str__(self):
        return str(self._config)

    def _update_trainer_kwargs(self):
        """
        This function will change the configuration parameters into objects accordingly to the attached dictionary
        """
        config_to_arg_dict = dict(
            foo="foo",
            early_stop_callback=EarlyStopping("val_loss")
        )

        for argument, included_in_training in self._config["trainer"].items():
            if included_in_training and argument in config_to_arg_dict:
                self._config["trainer"][argument] = config_to_arg_dict[argument]

    def _update_config_values(self):
        """
        This function will change configuration string values given for specific arguments into Python objects.
        """

        optimizer_dict = dict(
            SGD=SGD,
            Adam=Adam,
            AdamW=AdamW)

        loss_dict = dict(
            cross_entropy=nn.CrossEntropyLoss())

        model_dict = dict(
            mobilenet1=MobileNetV1,
            mobilenet2=MobileNetV2,
            mobilenet3_small=SmallMobileNetV3,
            mobilenet3_large=LargeMobileNetV3,
        )

        self._config["experiment"]["optimizer"] = optimizer_dict[self._config["experiment"]["optimizer"]]
        self._config["experiment"]["loss_function"] = loss_dict[self._config["experiment"]["loss_function"]]
        self._config["model"]["name"] = model_dict[self._config["model"]["name"]]

    def _log_config_in_logger(self):
        """
        This function is passing all analysis arguments to the logger in a specially formatted way
        :param config: dictionary with all arguments
        """
        self.log.info(f"\n\n{50 * '#'} Experiment configuration {50 * '#'}")
        for section, dct in self._config.items():
            self.log.info(f"\n{'-'*40} {section} {'-'*20}")
            for parameter, value in dct.items():
                self.log.info(f"{parameter}: {value}")
        self.log.info(f"{'#' * 100}")
