import argparse
import logging
from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
from Data_Preparation.Tac import tac
import torch
import numpy as np
from torch import Tensor
import os
from utils import load_data_inlier, load_data_test, load_hyperparams, get_data_fasttext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
    names = ['enron', 'mage']
    inlier_topics = ['normal']


    for name in names:
        print(name)
        for inlier_topic in inlier_topics:
            print(inlier_topic)
            data_train = load_data_inlier(name, inlier_topic, save_dir, is_infec=False, is_cvdd=True)
            X_inlier = Tensor(data_train['sbert_embeddings']).to(device)
            print(X_inlier.shape)

            for n_run in range(1, 11):
                data_test = load_data_test(name, inlier_topic, n_run, save_dir, is_cvdd=True)
                y_test = Tensor(data_test['anomaly_class'])
                X_test = Tensor(data_test['sbert_embeddings']).to(device)
                if n_run == 4:
                    print(y_test.shape, X_test.shape)
                    print(np.unique(y_test, return_counts=True))
                    print("-------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="20newsgroups"
    )

    parser.add_argument(
        "--inlier_topic",
        type=str,
        default="computer"
    )

    parser.add_argument(
        "--type_tac",
        type=str,
        default="ruff"
    )

    parser.add_argument(
        "--nu",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--type_encoder",
        type=str,
        default="sentencebert"
    )

    parser.add_argument(
        "--model_encoder",
        type=str,
        default="all-distilroberta-v1"
    )

    parser.add_argument(
        "--whichset",
        type=str,
        default="all"
    )

    parser.add_argument(
        "--nbruns",
        type=int,
        default=10
    )

    args = parser.parse_args()
    main(args)