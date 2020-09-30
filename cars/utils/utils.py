"""
Useful functions used across different modules
"""

import logging
import tarfile
import zipfile
from datetime import datetime

import numpy as np
import torch
from fvcore.nn.flop_count import flop_count

from cars.config.structure import get_project_structure


def convert_tar_to_zip(tar_archive_path_or_stream, tar_archive_open_mode, zip_archive_path, delete=True):
    tar_archive = tarfile.open(name=tar_archive_path_or_stream, mode=tar_archive_open_mode)
    zip_archive = zipfile.ZipFile(file=zip_archive_path, mode='a', compression=zipfile.ZIP_DEFLATED)
    for file_to_copy in tar_archive:
        file = tar_archive.extractfile(file_to_copy)
        if file:
            file_name = file_to_copy.name.split("/")[-1]
            file_content = file.read()
            zip_archive.writestr(file_name, file_content)
    tar_archive.close()
    zip_archive.close()

    if delete:
        tar_archive_path_or_stream.unlink()


def configure_default_logging(name):

    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    logging_dir = get_project_structure()['logging_dir']
    today = datetime.today().strftime('%Y-%m-%d')

    logging_dir.mkdir(exist_ok=True, parents=True)
    log_file = f'{today}.log'

    file_handler = logging.FileHandler(logging_dir / log_file)
    file_handler.setLevel(logging.DEBUG)
    file_form = f'%(asctime)s: %(name)s: %(levelname)s: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_form))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_form = f'%(asctime)s: %(name)s: %(message)s'
    console_handler.setFormatter(logging.Formatter(console_form))

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    log.addHandler(console_handler)

    return log


def calculate_model_stats(model, image_size):
    example_batch_input = torch.rand([1, 3, image_size, image_size]).to(model.device)
    flop_results = flop_count(model, (example_batch_input,))

    total_params = np.sum([p.numel() for p in model.parameters()]).item()
    total_flops = sum(flop_results[0].values())

    return total_params, total_flops
