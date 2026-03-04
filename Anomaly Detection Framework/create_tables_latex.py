import argparse
import logging
from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
import torch
import os
from utils import generate_tables_for_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_topics_dict= {
    '20newsgroups' : ['computer', 'recreation', 'science', 'miscellaneous', 'politics', 'religion'],
    'reuters' : ['earn', 'trade', 'acq', 'money-fx', 'crude', 'ship', 'interest'],
    'agnews' : ['World', 'Sports', 'Business', 'Sci-Tech'] ,
    'dbpedia14' : ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", 
                  "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work"]
}

def main(args):

    if args.all_embeddings:
        encoding_list = ['sentence-bert', 'bert', 'fasttext']
    else:
        encoding_list = [args.type_embedding]

    if args.all_nus:
        nu_list = [0.0, 0.05, 0.1, 0.15, 0.2]
    else:
        nu_list = [args.nu]

    if args.all_metrics:
        metrics = ['auc', 'fpr95', 'ap']
    else:
        metrics = [args.metric]

    for encoding in encoding_list:
        for nu in nu_list:
            # generate_tables_for_config(encoding, nu)
            for me in metrics:
                print(f"<<<< {encoding}, {nu}, {me} >>>> \n")
                generate_tables_for_config("sentence-bert", 0.0, metric=me)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Tables Latex script")

    parser.add_argument(
        "--all_embeddings",
        action="store_true"
    )

    args, remaining_argv = parser.parse_known_args()

    if not args.all_embeddings:
        parser.add_argument(
        "--type_embedding",
        type=str,
        default="sentence-bert"
        )

    parser.add_argument(
        "--all_nus",
        action="store_true"
    )

    args, remaining_argv = parser.parse_known_args()

    if not args.all_nus:
        parser.add_argument(
        "--nu",
        type=float,
        default=0.0
        )

    
    parser.add_argument(
        "--all_metrics",
        action="store_true"
    )

    args, remaining_argv = parser.parse_known_args()

    if not args.all_metrics:
        parser.add_argument(
        "--metric",
        type=str,
        default="auc"
        )

    args = parser.parse_args()
    main(args)