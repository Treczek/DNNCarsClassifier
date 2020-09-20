"""
Module containing Dataset with Stanford train and test images
"""

import pandas as pd
import requests
import torch

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from zipfile import ZipFile

from cars.config import get_data_sources
from cars.utils import convert_tar_to_zip


class StanfordCarsDataset(Dataset):

    def __init__(self, mode, transformer, logger, image_size=(227, 227)):
        super().__init__()

        self.log = logger

        self.mode = mode
        self.log.info(f"Preparation of {mode} dataset started.")
        self.transformer = transformer
        self.image_size = image_size

        self.sources = get_data_sources()
        self.tgz_data_path = self.sources["stanford"][self.mode]["location"]
        self.zip_data_path = self.tgz_data_path.parent.resolve() / f"cars_{self.mode}.zip"

        if not self.zip_data_path.exists() and not self.tgz_data_path.exists():
            self.log.info(f"{self.mode} dataset not found - downloading...")
            self._download_dataset()

        if not self.zip_data_path.exists() and self.tgz_data_path.exists():
            self.log.info(f"Converting {self.mode} tgz archive into zip file")
            convert_tar_to_zip(tar_archive_path_or_stream=self.tgz_data_path,
                               tar_archive_open_mode='r|gz',
                               zip_archive_path=self.zip_data_path,
                               delete=False)

        self.zipped_data = ZipFile(self.zip_data_path)

        self.image_file_names = self._get_file_names()
        self.labels = self._get_labels()

    def _download_dataset(self):

        source = self.sources["stanford"][self.mode]["source"]

        buffer = BytesIO(requests.get(source).content)
        convert_tar_to_zip(tar_archive_path_or_stream=buffer,
                           tar_archive_open_mode='rb|gz',
                           zip_archive_path=self.zip_data_path,
                           delete=False)
        self.log.info(f"{self.mode} dataset downloaded and saved.")

    def _get_file_names(self):
        file_names = [img.filename for img in self.zipped_data.filelist]
        self.log.info("File names successfully obtained from the zip archive")
        return file_names

    def _get_labels(self):
        # Depending on existence of file with labels we will use path to local file or byte stream from given url
        stream_or_path = self.sources["stanford"]["labels"]["location"]
        file_url = self.sources["stanford"]["labels"]["source"]

        if not stream_or_path.exists():
            r = requests.get(file_url)
            stream_or_path = BytesIO(r.content)
            self.log.info("Labels downloaded from the source url")

        labels = (pd
                  .read_csv(stream_or_path, usecols=['filename', 'class_id', 'is_test'])
                  .query(f'is_test == {(self.mode == "test")}')
                  .drop(columns=["is_test"]))

        labels["class_id"] -= 1  # classes starts from 1 instead of 0
        if self.mode == "test":
            labels["filename"] = labels["filename"].str.replace("test_", "")

        self.log.info(f"{self.mode} labels successfully loaded.")

        return labels

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        image_name = self.image_file_names[idx]
        img = Image.open(self.zipped_data.open(image_name)).convert('RGB')
        img = self.transformer(img)

        mask = self.labels["filename"] == image_name
        label = torch.as_tensor(self.labels[mask]["class_id"].item())

        return img, label
