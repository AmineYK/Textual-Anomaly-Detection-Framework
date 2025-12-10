#!/bin/bash

# cd "$(dirname "$0")" || exit 1


# python3 run_indep_anom.py --dataset_name reuters --inlier_topic interest --fm
python3 run_indep_anom.py --dataset_name agnews --runall --rsrae --ae
# python3 create_data_matrix.py --dataset_name dbpedia14 --type_tac pantin --nu 0.1 \
#                 --type_encoder sentencebert --model_encoder all-distilroberta-v1 \
#                 --whichset test --nbruns 10

























# python3 run_contex_anom.py --dataset_name 20newsgroups --inlier_topic computer --type_tac pantin \
#                     --nu 0.1 --type_encoder sentencebert --model_encoder all-distilroberta-v1 \
#                     --whichset test --nbruns 10

# python3 main.py --dataset_name 20newsgroups --training_mode one_class --device 'cuda' --preprocessing --inlier_topic computer \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model fm 


# python3 main.py --dataset_name 20newsgroups --training_mode one_class --device 'cuda' --preprocessing --inlier_topic computer \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model cvdd --attention_size 150 --n_attention_heads 10 \
#     --lambda_p 1.0 --alpha_scheduler "logarithmic" --n_epochs 5 --lr 0.01 --lr_milestones 2 3

# python3 main_all_runs.py --dataset_name reuters --training_mode one_class --inlier_topic earn \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 0.1

# python3 main.py --dataset_name reuters --training_mode one_class --inlier_topic acq \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 1

# python3 main.py --dataset_name reuters --training_mode one_class --inlier_topic crude \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 1

# python3 main.py --dataset_name reuters --training_mode one_class --inlier_topic trade \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 1

# python3 main.py --dataset_name reuters --training_mode one_class --inlier_topic money-fx \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 1

# python3 main.py --dataset_name reuters --training_mode one_class --inlier_topic interest \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 1

# python3 main.py --dataset_name reuters --training_mode one_class --inlier_topic ship \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 1

###################################
######### 20NewsGroups ############
###################################

# python3 main.py --dataset_name 20newsgroups --training_mode one_class --inlier_topic science \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 64 --shuffle --ad_model ocsvm --nu 0.05 --kernel 'rbf' --gamma 1


# python3 main.py --dataset_name reuters --training_mode one_class --inlier_topic acq \
# --type_tac ruff --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
# --batch_size 64 --shuffle --ad_model cvdd --attention_size 150 --n_attention_heads 10 

# python3 main.py --dataset_name 20NewsGroups --training_mode two_classes --inlier_topic science \
#     --type_tac ruff --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf 

# python3 main.py --dataset_name WOS --training_mode two_classes --inlier_topic Civil_Engineering \
#     --type_tac pantin --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf

# python3 main.py --dataset_name DBpedia14 --training_mode two_classes --inlier_topic Animal \
#     --type_tac pantin --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf

# python3 main.py --dataset_name AGNews --training_mode two_classes --inlier_topic Sports \
#     --type_tac fate --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf


# python3 main.py --dataset_name Reuters --training_mode two_classes --inlier_topic energy \
#     --type_tac pantin --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 32 --shuffle

# python3 main.py --dataset_name AGNews --inlier_topic Sports --type_tac fate \
#     --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf

# python3 main.py --dataset_name AGNews --inlier_topic Sports --type_tac fate \
#     --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove

# python3 main.py --dataset_name AGNews --inlier_topic Sports --type_tac fate \
#     --anomaly_rate 0.1 --emb_model fasttext_300d.kv --type_emb fasttext


# python3 main.py --dataset_name Reuters --training_mode one_class --inlier_topic energy \
#     --type_tac pantin --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf \
#     --batch_size 32 --shuffle

# python3 main.py --dataset_name Reuters --training_mode two_classes --inlier_topic energy \
#     --type_tac pantin --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf \
#     --batch_size 32 --shuffle

# python3 main.py --dataset_name WOS --training_mode one_class --inlier_topic Civil_Engineering \
#     --type_tac pantin --anomaly_rate 0.1 --emb_model tfidf --type_emb tfidf \
#     --batch_size 32 --shuffle --ad_model ocsvm

# python3 main.py --dataset_name Reuters --training_mode one_class --inlier_topic energy \
#     --type_tac pantin --anomaly_rate 0.1 --emb_model glove_300d.kv --type_emb glove \
#     --batch_size 32 --shuffle --ad_model cvdd --attention_size 150 --n_attention_heads 2
