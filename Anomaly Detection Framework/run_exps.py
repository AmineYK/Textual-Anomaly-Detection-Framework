import argparse
import logging
# from Data_Preparation import utils
from Data_Preparation.Embedding import embedding_encoder
from utils import load_data_inlier, load_data_test, load_hyperparams

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

    # if we'll run one inlier category or all categories for the dataset
    if args.runall:
        inlier_topics = dataset_topics_dict[args.dataset_name]
    else:
        inlier_topics = [args.inlier_topic]

    save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
    file_path_hyp = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results/hyperparams.txt"


    # for every inlier category 
    for inlier_topic in inlier_topics:

        if args.fm:
            list_auc_fm = []
            list_fpr_fm = []
            list_ap_fm = []
            list_time_fm = []

        if args.ocsvm:
            list_auc_ocsvm = []
            list_fpr_ocsvm = []
            list_ap_ocsvm = []
            list_time_ocsvm = []

        if args.rsrae:
            list_auc_rsrae = []
            list_fpr_rsrae = []
            list_ap_rsrae = []
            list_time_rsrae = []

        if args.ae:
            list_auc_ae = []
            list_fpr_ae = []
            list_ap_ae = []
            list_time_ae = []

        if args.tccm:
            list_auc_tccm = []
            list_fpr_tccm = []
            list_ap_tccm = []
            list_time_tccm = []

        # load the X_inlier matrix
        X_inlier = load_data_inlier(args.dataset_name, inlier_topic, save_dir)
        print(X_inlier.shape)
        # get the hyperparamter for the FM model for this specific inlier category
        hyp = load_hyperparams(args.dataset_name, inlier_topic, file_path_hyp)

        for n_run in range(1,11):

            X_test, y_test = load_data_test(args.dataset_name, inlier_topic, n_run, save_dir)
            print(X_test.shape, y_test.shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="20newsgroups"
    )

    parser.add_argument(
        "--runall",
        action="store_true"
    )

    args, remaining_argv = parser.parse_known_args()

    if not args.runall:
        parser.add_argument(
        "--inlier_topic",
        type=str,
        default="computer"
        )

    parser.add_argument(
    "--fm",
    action="store_true"
    )

    parser.add_argument(
    "--ocsvm",
    action="store_true"
    )

    parser.add_argument(
    "--ae",
    action="store_true"
    )

    parser.add_argument(
    "--rsrae",
    action="store_true"
    )   

    parser.add_argument(
    "--tccm",
    action="store_true"
    )  
    

    args = parser.parse_args()
    main(args)