import time
from datasets import concatenate_datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import numpy as np
from torch.utils.data import DataLoader

# Filtring documents per class (20NewsGroups)
#--------------------------------------------

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

def textual_anomaly_contamination(dataset, inlier_topic, anomaly_type, nb_documents = 100, anomaly_rate = 0.1, batch_size=32, seed=42): 

    intlier_subset, anoamlie_subset = get_documents_from_class(dataset.dataset, inlier_topic, compl=True)

    # must verify that the number of document for this category >= nb_documents * (1 - anomaly_rate)
    assert intlier_subset.num_rows >= int(nb_documents * (1 - anomaly_rate)), f"There is not enough documents in the requested category ({inlier_topic}). Please reduce the number of documents to at least {intlier_subset.num_rows}"

    if anomaly_type == "independent":
        type_anomalie_subset = anoamlie_subset.filter(lambda x: x["label_text"].split(".")[0] != inlier_topic.split(".")[0])

    elif anomaly_type == "contexual":
        type_anomalie_subset = anoamlie_subset.filter(
    lambda x: (
        (x["label_text"].split(".")[0] == inlier_topic.split(".")[0]) and
        (x["label_text"].split(".")[1] != inlier_topic.split(".")[1])
      )
    ) 


    nb_anomaly_samples = int(nb_documents * anomaly_rate)
    nb_inlier_samples = nb_documents - nb_anomaly_samples

    # selecting indices 
    anom_indices = np.random.randint(0,type_anomalie_subset.num_rows,nb_anomaly_samples)
    final_anomaly_subset = type_anomalie_subset.select(anom_indices)
    final_anomaly_subset = final_anomaly_subset.map(add_col,fn_kwargs={"anomaly_class": 1} )


    inlier_indices = np.random.randint(0,intlier_subset.num_rows,nb_inlier_samples)
    final_inlier_subset = intlier_subset.select(inlier_indices)
    final_inlier_subset = final_inlier_subset.map(add_col,fn_kwargs={"anomaly_class": 0} )

    TAC_dataset = concatenate_datasets([final_inlier_subset, final_anomaly_subset]).shuffle(seed=seed)


    # verifications
    assert TAC_dataset.num_rows == nb_documents, "The right number of documents is not selected"
    assert TAC_dataset['anomaly_label'].count(1) == int(nb_documents * anomaly_rate), "The right number of anomaly documents is not selected"
    assert TAC_dataset['anomaly_label'].count(0) == nb_documents - int(nb_documents * anomaly_rate), "The right number of inlier documents is not selected"

    # in the inlier subset : there is no text with different 'inlier_topic' label
    assert (TAC_dataset.filter(lambda x: x['anomaly_label'] == 0).filter(lambda x: x['label_text'] != inlier_topic)).num_rows == 0, "There is document in the inlier subset with wrong label"

    if anomaly_type == "independent":
        assert (TAC_dataset.filter(lambda x: x['anomaly_label'] == 1)
    .filter(lambda x: x["label_text"].split(".")[0] == inlier_topic.split(".")[0])).num_rows == 0, "The independ anomalies are not well constructed"

    if anomaly_type == "contexual":
        assert (TAC_dataset.filter(lambda x: x['anomaly_label'] == 1)
    .filter(lambda x: (x["label_text"].split(".")[0] == inlier_topic.split(".")[0]) and
          (x["label_text"].split(".")[1] != inlier_topic.split(".")[1]))).num_rows == int(nb_documents * anomaly_rate), "The contexual anomalies are not well constructed"


    return DataLoader(TAC_dataset, batch_size=batch_size, shuffle=True)  



# Textual Anomaly Contamination TAC (Reuters Ruff)
#-------------------------------------------------

def textual_anomaly_contamination_reuters(dataset, inlier_topic, type_tac='ruff', batch_size=64):
    
    # same strategy for both r'ruff' and 'pantin' method
    dataset_one_label = dataset.filter( lambda x : len(x['topics']) == 1)
        
    #############################################
    ################## RUFF  ####################
    #############################################
    
    
    if type_tac == 'ruff':
        values, counts = np.unique(dataset_one_label[:]['topics'],return_counts=True)
        selected_labels = values[counts >= 100]

        dataset_ruff = dataset_one_label.filter(lambda x : x['topics'] in selected_labels)

    
        if inlier_topic not in selected_labels:
            raise Exception(" Warning ! the inlier topic requested does't exsite with this tac methode !")
        else:  
            intlier_dataset_ruff = dataset_ruff.filter(lambda x : x['topics'] == [inlier_topic])
            anomaly_dataset_ruff = dataset_ruff.filter(lambda x : x['topics'] != [inlier_topic])

            assert intlier_dataset_ruff.num_rows + anomaly_dataset_ruff.num_rows == dataset_ruff.num_rows, "some samples of original dataset are missing in the training and the testing dataset" 

            return DataLoader(intlier_dataset_ruff, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_ruff, batch_size=batch_size, shuffle=True) 

    
    
    #############################################
    ################# PANTIN ####################
    #############################################
    
    
    if type_tac == 'pantin':
        parent_topics = {
            "commodities": [
                "acq", "carcass", "cocoa", "coconut", "coffee", "cotton",
                "grain", "groundnut", "hog", "housing", "l-cattle", "livestock",
                "lumber", "meal-feed", "oilseed", "orange", "potato", "rice",
                "rubber", "ship", "sugar", "tea", "veg-oil"
            ],

            "financial": [
                "cpi", "cpu", "dlr", "earn", "income", "instal-debt", "interest",
                "ipi", "jobs", "lei", "money-fx", "money-supply", "rand",
                "reserves", "retail", "trade", "wpi", "yen"
            ],

            "metals": [
                "alum", "copper", "gold", "iron-steel", "lead", "nickel",
                "platinum", "silver", "strategic-metal", "tin", "zinc"
            ],

            "energy": [
                "crude", "fuel", "heat", "jet", "naphtha", "nat-gas", "propane"
            ]
        }
        
        topic_map = {topic: parent for parent, topics in parent_topics.items() for topic in topics}

        def add_parent_topic(row):

            row["parent_topic"] = topic_map.get(row["topics"][0], "unknown")

            return row

        dataset_pantin = dataset_one_label.map(add_parent_topic)
        
        if inlier_topic not in parent_topics.keys():
            raise Exception(" Warning ! the inlier topic requested does't exsite with this tac methode !")           
        else:  
            intlier_dataset_pantin = dataset_pantin.filter(lambda x : x['parent_topic'] == inlier_topic)
            anomaly_dataset_pantin = dataset_pantin.filter(lambda x : x['parent_topic'] != inlier_topic)

            assert intlier_dataset_pantin.num_rows + anomaly_dataset_pantin.num_rows == dataset_pantin.num_rows, "some samples of original dataset are missing in the training and the testing dataset" 

            return DataLoader(intlier_dataset_pantin, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_pantin, batch_size=batch_size, shuffle=True) 

    raise Exception(" the 'type_tac' selected is not available for this dataset ")
    
    
# Textual Anomaly Contamination TAC (20NewsGroups Ruff & Pantin)
#-----------------------------------------------------

def textual_anomaly_contamination_20newsgroups(dataset, inlier_topic, type_tac='ruff', batch_size=64):
   
    
    if type_tac == 'ruff':
        
        groups = {
            "computer": [
                "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware", "comp.windows.x"
            ],
            "recreation": [
                "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"
            ],
            "science": [
                "sci.crypt", "sci.electronics", "sci.med", "sci.space"
            ],
            "miscellaneous": [
                "misc.forsale"
            ],
            "politics": [
                "talk.politics.misc", "talk.politics.guns", "talk.politics.mideast"
            ],
            "religion": [
                "talk.religion.misc", "alt.atheism", "soc.religion.christian"
            ]
        }
        
    elif type_tac == 'pantin':
        
        groups = {
            "computer": [
                "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware", "comp.windows.x"
            ],
            "motors": [
                "rec.motorcycles", "rec.autos"
            ],
            "sports": [
                 "rec.sport.baseball", "rec.sport.hockey"
            ],
            "science": [
                "sci.crypt", "sci.electronics", "sci.med", "sci.space"
            ],
            "forsale": [
                "misc.forsale"
            ],
            "politics": [
                "talk.politics.misc", "talk.politics.guns", "talk.politics.mideast"
            ],
            "religion": [
                "talk.religion.misc", "alt.atheism", "soc.religion.christian"
            ]
        }
        
    else: raise Exception(" the 'type_tac' selected is not available for this dataset ")

    topic_map = {label: topic for topic, labels in groups.items() for label in labels}

    def add_topic_label(row):
        row["topic_label_text"] = topic_map.get(row["label_text"], "unknown")
        return row

    dataset = dataset.map(add_topic_label)

    if inlier_topic not in list(groups.keys()):

        raise Exception(" Warning ! the inlier topic requested does't exsite with this tac methode !")

    else:

        intlier_dataset_ruff = dataset.filter(lambda x : x['topic_label_text'] == inlier_topic)
        intlier_dataset_ruff = intlier_dataset_ruff.map(add_col,fn_kwargs={"anomaly_class": 0} )

        anomaly_dataset_ruff = dataset.filter(lambda x : x['topic_label_text'] != inlier_topic)
        anomaly_dataset_ruff = anomaly_dataset_ruff.map(add_col,fn_kwargs={"anomaly_class": 1} )

        assert intlier_dataset_ruff.num_rows + anomaly_dataset_ruff.num_rows == dataset.num_rows, "some samples of original dataset are missing in the training and the testing dataset" 


    return DataLoader(intlier_dataset_ruff, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_ruff, batch_size=batch_size, shuffle=True) 



# Textual Anomaly Contamination TAC (WOS Pantin)
#-----------------------------------------------------

def textual_anomaly_contamination_wos(dataset, inlier_topic, type_tac='pantin', batch_size=64):
    
    ###############################################
    ################## PANTIN  ####################
    ###############################################
    
    if type_tac == 'pantin':
        
        level_1_mapping = {
            "Computer_Science" : 0,
            "Electrical_Engineering" : 1,
            "Psychology" : 2,
            "Mechanical_Engineering" : 3,
            "Civil_Engineering" : 4,
            "Medical_Science" : 5,
            "Biochemistry" : 6
        }
        

        if inlier_topic not in list(level_1_mapping.keys()):

            raise Exception(" Warning ! the inlier topic requested does't exsite with this tac methode !")

        else:
            
            # inlier_topic_int = np.array(list(level_1_mapping.keys()))[np.array(list(level_1_mapping.values())) == inlier_topic][0]

            intlier_dataset_pantin = dataset.filter(lambda x: x['label_level_1'] == level_1_mapping[inlier_topic])
            intlier_dataset_pantin = intlier_dataset_pantin.map(add_col,fn_kwargs={"anomaly_class": 0} )

            anomaly_dataset_pantin = dataset.filter(lambda x: x['label_level_1'] != level_1_mapping[inlier_topic])
            anomaly_dataset_pantin = anomaly_dataset_pantin.map(add_col,fn_kwargs={"anomaly_class": 1} )

        
            assert intlier_dataset_pantin.num_rows + anomaly_dataset_pantin.num_rows == dataset.num_rows, "some samples of original dataset are missing in the training and the testing dataset" 


        return DataLoader(intlier_dataset_pantin, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_pantin, batch_size=batch_size, shuffle=True) 
    
    raise Exception(" the 'type_tac' selected is not available for this dataset ")
    
    
# Textual Anomaly Contamination TAC (DBedia 14 Pantin)
#-----------------------------------------------------

def textual_anomaly_contamination_dbedia14(dataset, inlier_topic, type_tac='pantin', batch_size=64):
    
    ###############################################
    ################## PANTIN  ####################
    ###############################################
    
    if type_tac == 'pantin':
        
        level_1_mapping = {
            "Company": 0,
            "Educational Institution": 1,
            "Artist": 2,
            "Athlete": 3,
            "Office Holder": 4,
            "Mean Of Transportation": 5,
            "Building": 6,
            "Natural Place": 7,
            "Village": 8,
            "Animal": 9,
            "Plant": 10,
            "Album": 11,
            "Film": 12,
            "Written Work": 13
        }
        

        if inlier_topic not in list(level_1_mapping.keys()):

            raise Exception(" Warning ! the inlier topic requested does't exsite with this tac methode !")

        else:
            
            # inlier_topic_int = np.array(list(level_1_mapping.keys()))[np.array(list(level_1_mapping.values())) == inlier_topic][0]

            intlier_dataset_pantin = dataset.filter(lambda x: x['label'] == level_1_mapping[inlier_topic])
            intlier_dataset_pantin = intlier_dataset_pantin.map(add_col,fn_kwargs={"anomaly_class": 0} )

            anomaly_dataset_pantin = dataset.filter(lambda x: x['label'] != level_1_mapping[inlier_topic])
            anomaly_dataset_pantin = anomaly_dataset_pantin.map(add_col,fn_kwargs={"anomaly_class": 1} )

        
            assert intlier_dataset_pantin.num_rows + anomaly_dataset_pantin.num_rows == dataset.num_rows, "some samples of original dataset are missing in the training and the testing dataset" 


        return DataLoader(intlier_dataset_pantin, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_pantin, batch_size=batch_size, shuffle=True) 
    
    raise Exception(" the 'type_tac' selected is not available for this dataset ")
    
    
# Textual Anomaly Contamination TAC (DBedia 14 Pantin)
#-----------------------------------------------------

def textual_anomaly_contamination_agnews(dataset, inlier_topic, type_tac='fate', batch_size=64):
    
    ###############################################
    ################## PANTIN  ####################
    ###############################################
    
    if type_tac == 'fate':
        
        mapping = {
            "World" : 0,
            "Sports" : 1,
            "Business" : 2,
            "Sci/Tech" : 3
        }


        if inlier_topic not in list(mapping.keys()):

            raise Exception(" Warning ! the inlier topic requested does't exsite with this tac methode !")

        else:
            
            # inlier_topic_int = np.array(list(level_1_mapping.keys()))[np.array(list(level_1_mapping.values())) == inlier_topic][0]

            intlier_dataset_fate = dataset.filter(lambda x: x['label'] == mapping[inlier_topic])
            intlier_dataset_fate = intlier_dataset_fate.map(add_col,fn_kwargs={"anomaly_class": 0} )

            anomaly_dataset_fate = dataset.filter(lambda x: x['label'] != mapping[inlier_topic])
            anomaly_dataset_fate = anomaly_dataset_fate.map(add_col,fn_kwargs={"anomaly_class": 1} )

        
            assert intlier_dataset_fate.num_rows + anomaly_dataset_fate.num_rows == dataset.num_rows, "some samples of original dataset are missing in the training and the testing dataset" 


        return DataLoader(intlier_dataset_fate, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_fate, batch_size=batch_size, shuffle=True) 
    
    
    raise Exception(" the 'type_tac' selected is not available for this dataset ")