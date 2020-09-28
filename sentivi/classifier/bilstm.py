import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import classification_report

from sentivi.classifier.nn_clf import NeuralNetworkDataset
from sentivi.classifier.nn_clf import NeuralNetworkClassifier


class BiLSTM(nn.Module):
    def __init__(self, num_labels: int, embedding_size: int, max_length: int):
        """
        Initialize BiLSTM classifier
        :param num_labels:
        :param embedding_size:
        :param max_length:
        """
        super(BiLSTM, self).__init__()

    def forward(self, x):
        """
        Forward method for torch.nn.Module
        :param x:
        :return:
        """

        return x
