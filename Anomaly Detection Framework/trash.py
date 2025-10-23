
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
