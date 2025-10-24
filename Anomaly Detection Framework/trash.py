
  # *****************************
    if name == "Enron":

        dataset = load_dataset("corbt/enron-emails")

        return DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
  
  # *****************************
    if name == "SMS Sp.":
    
        dataset =  load_dataset("ucirvine/sms_spam")

        return DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    
    
    
  # *****************************
    if name == "SST2":

        dataset =  load_dataset("rungalileo/sst2")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader, validation_dataloader
    
    # *****************************
    if name == "IMDB":

        dataset =  load_dataset("stanfordnlp/imdb")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        unsupervised_dataloader = DataLoader(dataset['unsupervised'], batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader, unsupervised_dataloader
    
    
    
    

#     #############################################
#     ################# PANTIN ####################
#     #############################################
    
#     if type_tac == 'pantin':
        
#         groups = {
#             "computer": [
#                 "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
#                 "comp.sys.mac.hardware", "comp.windows.x"
#             ],
#             "motors": [
#                 "rec.motorcycles", "rec.autos"
#             ],
#             "sports": [
#                  "rec.sport.baseball", "rec.sport.hockey"
#             ],
#             "science": [
#                 "sci.crypt", "sci.electronics", "sci.med", "sci.space"
#             ],
#             "forsale": [
#                 "misc.forsale"
#             ],
#             "politics": [
#                 "talk.politics.misc", "talk.politics.guns", "talk.politics.mideast"
#             ],
#             "religion": [
#                 "talk.religion.misc", "alt.atheism", "soc.religion.christian"
#             ]
#         }

#         topic_map = {label: topic for topic, labels in groups.items() for label in labels}

#         def add_topic_label(row):
#             row["topic_label_text"] = topic_map.get(row["label_text"], "unknown")
#             return row

#         dataset = dataset.map(add_topic_label)
        
#         if inlier_topic not in list(groups.keys()):
        
#             raise Exception(" Warning ! the inlier topic requested does't exsite with this tac methode !")
    
#         else:
        
#             intlier_dataset_ruff = dataset.filter(lambda x : x['topic_label_text'] == inlier_topic)
#             intlier_dataset_ruff = intlier_dataset_ruff.map(add_col,fn_kwargs={"anomaly_class": 0} )
                
#             anomaly_dataset_ruff = dataset.filter(lambda x : x['topic_label_text'] != inlier_topic)
#             anomaly_dataset_ruff = anomaly_dataset_ruff.map(add_col,fn_kwargs={"anomaly_class": 1} )
             
#             assert intlier_dataset_ruff.num_rows + anomaly_dataset_ruff.num_rows == dataset.num_rows, "some samples of original dataset are missing in the training and the testing dataset" 
            
            
#         return DataLoader(intlier_dataset_ruff, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_ruff, batch_size=batch_size, shuffle=True) 



def get_documents_from_class(dataset, class_name="comp.graphics", compl=True, verbose=False):

    tac = time.time()

    subset_class = dataset.filter(lambda x: x["label_text"] == class_name)
    if compl:
        subset_compl = dataset.filter(lambda x: x["label_text"] != class_name)

    tic = time.time()

    if verbose:
        print(f"Documents of '{class_name}' class extracted in {np.round(tic-tac,2)}s\n\n")

    if compl: return subset_class, subset_compl
    return subset_class


def add_col(example,anomaly_class = 0):
    
    example["anomaly_label"] = anomaly_class
    return example


# Textual Anomaly Contamination TAC (20NewsGroups)
#-------------------------------------------------

# def textual_anomaly_contamination(dataset, inlier_topic, anomaly_type, nb_documents = 100, anomaly_rate = 0.1, batch_size=32, seed=42): 

#     intlier_subset, anoamlie_subset = get_documents_from_class(dataset.dataset, inlier_topic, compl=True)

#     # must verify that the number of document for this category >= nb_documents * (1 - anomaly_rate)
#     assert intlier_subset.num_rows >= int(nb_documents * (1 - anomaly_rate)), f"There is not enough documents in the requested category ({inlier_topic}). Please reduce the number of documents to at least {intlier_subset.num_rows}"

#     if anomaly_type == "independent":
#         type_anomalie_subset = anoamlie_subset.filter(lambda x: x["label_text"].split(".")[0] != inlier_topic.split(".")[0])

#     elif anomaly_type == "contexual":
#         type_anomalie_subset = anoamlie_subset.filter(
#     lambda x: (
#         (x["label_text"].split(".")[0] == inlier_topic.split(".")[0]) and
#         (x["label_text"].split(".")[1] != inlier_topic.split(".")[1])
#       )
#     ) 


#     nb_anomaly_samples = int(nb_documents * anomaly_rate)
#     nb_inlier_samples = nb_documents - nb_anomaly_samples

#     # selecting indices 
#     anom_indices = np.random.randint(0,type_anomalie_subset.num_rows,nb_anomaly_samples)
#     final_anomaly_subset = type_anomalie_subset.select(anom_indices)
#     final_anomaly_subset = final_anomaly_subset.map(add_col,fn_kwargs={"anomaly_class": 1} )


#     inlier_indices = np.random.randint(0,intlier_subset.num_rows,nb_inlier_samples)
#     final_inlier_subset = intlier_subset.select(inlier_indices)
#     final_inlier_subset = final_inlier_subset.map(add_col,fn_kwargs={"anomaly_class": 0} )

#     TAC_dataset = concatenate_datasets([final_inlier_subset, final_anomaly_subset]).shuffle(seed=seed)


#     # verifications
#     assert TAC_dataset.num_rows == nb_documents, "The right number of documents is not selected"
#     assert TAC_dataset['anomaly_label'].count(1) == int(nb_documents * anomaly_rate), "The right number of anomaly documents is not selected"
#     assert TAC_dataset['anomaly_label'].count(0) == nb_documents - int(nb_documents * anomaly_rate), "The right number of inlier documents is not selected"

#     # in the inlier subset : there is no text with different 'inlier_topic' label
#     assert (TAC_dataset.filter(lambda x: x['anomaly_label'] == 0).filter(lambda x: x['label_text'] != inlier_topic)).num_rows == 0, "There is document in the inlier subset with wrong label"

#     if anomaly_type == "independent":
#         assert (TAC_dataset.filter(lambda x: x['anomaly_label'] == 1)
#     .filter(lambda x: x["label_text"].split(".")[0] == inlier_topic.split(".")[0])).num_rows == 0, "The independ anomalies are not well constructed"

#     if anomaly_type == "contexual":
#         assert (TAC_dataset.filter(lambda x: x['anomaly_label'] == 1)
#     .filter(lambda x: (x["label_text"].split(".")[0] == inlier_topic.split(".")[0]) and
#           (x["label_text"].split(".")[1] != inlier_topic.split(".")[1]))).num_rows == int(nb_documents * anomaly_rate), "The contexual anomalies are not well constructed"


#     return DataLoader(TAC_dataset, batch_size=batch_size, shuffle=True)  