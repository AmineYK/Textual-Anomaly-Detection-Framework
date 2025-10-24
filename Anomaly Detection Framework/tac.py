import time
from datasets import concatenate_datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import numpy as np
from torch.utils.data import DataLoader


############################################
############### ADD COLUMN #################
############################################

def add_col(example, anomaly_class):
    example["anomaly_class"] = anomaly_class
    return example


############################################
################## TAC  ####################
############################################

def textual_anomaly_contamination(dataloader, dataset_name, inlier_topic, type_tac, anomaly_rate=0.1, batch_size=64):
    
    dataset = dataloader.dataset
    
    if dataset_name == '20NewsGroups':
        return textual_anomaly_contamination_20newsgroups(dataset, inlier_topic, type_tac, anomaly_rate, batch_size)
        
    if dataset_name == 'Reuters':
        return textual_anomaly_contamination_reuters(dataset, inlier_topic, type_tac, anomaly_rate, batch_size)
    
    if dataset_name == 'WOS':
        return textual_anomaly_contamination_wos(dataset, inlier_topic, type_tac, anomaly_rate, batch_size)
    
    if dataset_name == 'DBpedia14':
        return textual_anomaly_contamination_dbpedia14(dataset, inlier_topic, type_tac, anomaly_rate, batch_size)
        
    if dataset_name == 'AGNews':
        return textual_anomaly_contamination_agnews(dataset, inlier_topic, type_tac, anomaly_rate, batch_size)
        
    raise Exception("'dataset_name' not found")


############################################
############ Reuters Ruff/Pantin ###########
############################################

def textual_anomaly_contamination_reuters(dataset, inlier_topic, type_tac='ruff', anomaly_rate=0.1, batch_size=64):
    
    dataset_one_label = dataset.filter(lambda x: len(x['topics']) == 1)
    
    #############################################
    ################## RUFF  ####################
    #############################################
    
    if type_tac == 'ruff':
        values, counts = np.unique(dataset_one_label[:]['topics'], return_counts=True)
        selected_labels = values[counts >= 100]
        dataset_ruff = dataset_one_label.filter(lambda x: x['topics'] in selected_labels)
    
        if inlier_topic not in selected_labels:
            raise Exception(" Warning ! the inlier topic requested doesn't exist with this TAC method !")
        
        inlier_dataset_ruff = dataset_ruff.filter(lambda x: x['topics'] == [inlier_topic])
        anomaly_dataset_ruff = dataset_ruff.filter(lambda x: x['topics'] != [inlier_topic])
            
        n_anomalies = int((anomaly_rate * inlier_dataset_ruff.num_rows) / (1 - anomaly_rate))
        anomaly_indices = np.random.randint(0, anomaly_dataset_ruff.num_rows, n_anomalies)
        anomaly_dataset_ruff = anomaly_dataset_ruff.select(anomaly_indices)
        
        inlier_dataset_ruff = inlier_dataset_ruff.map(add_col, fn_kwargs={"anomaly_class": 0})
        anomaly_dataset_ruff = anomaly_dataset_ruff.map(add_col, fn_kwargs={"anomaly_class": 1})

        return DataLoader(inlier_dataset_ruff, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_ruff, batch_size=batch_size, shuffle=True)

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
            raise Exception(" Warning ! the inlier topic requested doesn't exist with this TAC method !")           
            
        inlier_dataset_pantin = dataset_pantin.filter(lambda x: x['parent_topic'] == inlier_topic)
        anomaly_dataset_pantin = dataset_pantin.filter(lambda x: x['parent_topic'] != inlier_topic)
            
        n_anomalies = int((anomaly_rate * inlier_dataset_pantin.num_rows) / (1 - anomaly_rate))
        anomaly_indices = np.random.randint(0, anomaly_dataset_pantin.num_rows, n_anomalies)
        anomaly_dataset_pantin = anomaly_dataset_pantin.select(anomaly_indices)
        
        inlier_dataset_pantin = inlier_dataset_pantin.map(add_col, fn_kwargs={"anomaly_class": 0})
        anomaly_dataset_pantin = anomaly_dataset_pantin.map(add_col, fn_kwargs={"anomaly_class": 1})

        return DataLoader(inlier_dataset_pantin, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset_pantin, batch_size=batch_size, shuffle=True)
    
    raise Exception(" the 'type_tac' selected is not available for this dataset ")


############################################
############ 20Newsgroups ##################
############################################

def textual_anomaly_contamination_20newsgroups(dataset, inlier_topic, type_tac='ruff', anomaly_rate=0.1, batch_size=64):
    
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
    else:
        raise Exception(" the 'type_tac' selected is not available for this dataset ")

    topic_map = {label: topic for topic, labels in groups.items() for label in labels}

    def add_topic_label(row):
        row["topic_label_text"] = topic_map.get(row["label_text"], "unknown")
        return row

    dataset = dataset.map(add_topic_label)

    if inlier_topic not in list(groups.keys()):
        raise Exception(" Warning ! the inlier topic requested doesn't exist with this TAC method !")

    inlier_dataset = dataset.filter(lambda x: x['topic_label_text'] == inlier_topic)
    anomaly_dataset = dataset.filter(lambda x: x['topic_label_text'] != inlier_topic)
    
    n_anomalies = int((anomaly_rate * inlier_dataset.num_rows) / (1 - anomaly_rate))
    anomaly_indices = np.random.randint(0, anomaly_dataset.num_rows, n_anomalies)
    anomaly_dataset = anomaly_dataset.select(anomaly_indices)

    inlier_dataset = inlier_dataset.map(add_col, fn_kwargs={"anomaly_class": 0})
    anomaly_dataset = anomaly_dataset.map(add_col, fn_kwargs={"anomaly_class": 1})

    return DataLoader(inlier_dataset, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True)


############################################
################## WOS #####################
############################################

def textual_anomaly_contamination_wos(dataset, inlier_topic, type_tac='pantin', anomaly_rate=0.1, batch_size=64):
    
    if type_tac != 'pantin':
        raise Exception(" the 'type_tac' selected is not available for this dataset ")
    
    level_1_mapping = {
        "Computer_Science": 0,
        "Electrical_Engineering": 1,
        "Psychology": 2,
        "Mechanical_Engineering": 3,
        "Civil_Engineering": 4,
        "Medical_Science": 5,
        "Biochemistry": 6
    }
    
    if inlier_topic not in level_1_mapping:
        raise Exception(" Warning ! the inlier topic requested doesn't exist with this TAC method !")
    
    inlier_dataset = dataset.filter(lambda x: x['label_level_1'] == level_1_mapping[inlier_topic])
    anomaly_dataset = dataset.filter(lambda x: x['label_level_1'] != level_1_mapping[inlier_topic])
    
    n_anomalies = int((anomaly_rate * inlier_dataset.num_rows) / (1 - anomaly_rate))
    anomaly_indices = np.random.randint(0, anomaly_dataset.num_rows, n_anomalies)
    anomaly_dataset = anomaly_dataset.select(anomaly_indices)
    
    inlier_dataset = inlier_dataset.map(add_col, fn_kwargs={"anomaly_class": 0})
    anomaly_dataset = anomaly_dataset.map(add_col, fn_kwargs={"anomaly_class": 1})

    return DataLoader(inlier_dataset, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True)


############################################
############ DBPedia14 #####################
############################################

def textual_anomaly_contamination_dbpedia14(dataset, inlier_topic, type_tac='pantin', anomaly_rate=0.1, batch_size=64):
    
    if type_tac != 'pantin':
        raise Exception(" the 'type_tac' selected is not available for this dataset ")
    
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
    
    if inlier_topic not in level_1_mapping:
        raise Exception(" Warning ! the inlier topic requested doesn't exist with this TAC method !")

    inlier_dataset = dataset.filter(lambda x: x['label'] == level_1_mapping[inlier_topic])
    anomaly_dataset = dataset.filter(lambda x: x['label'] != level_1_mapping[inlier_topic])
    
    n_anomalies = int((anomaly_rate * inlier_dataset.num_rows) / (1 - anomaly_rate))
    anomaly_indices = np.random.randint(0, anomaly_dataset.num_rows, n_anomalies)
    anomaly_dataset = anomaly_dataset.select(anomaly_indices)
    
    inlier_dataset = inlier_dataset.map(add_col, fn_kwargs={"anomaly_class": 0})
    anomaly_dataset = anomaly_dataset.map(add_col, fn_kwargs={"anomaly_class": 1})

    return DataLoader(inlier_dataset, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True)


############################################
################## AGNews ##################
############################################

def textual_anomaly_contamination_agnews(dataset, inlier_topic, type_tac='fate', anomaly_rate=0.1, batch_size=64):
    
    if type_tac != 'fate':
        raise Exception(" the 'type_tac' selected is not available for this dataset ")
    
    mapping = {
        "World": 0,
        "Sports": 1,
        "Business": 2,
        "Sci/Tech": 3
    }
    
    if inlier_topic not in mapping:
        raise Exception(" Warning ! the inlier topic requested doesn't exist with this TAC method !")

    inlier_dataset = dataset.filter(lambda x: x['label'] == mapping[inlier_topic])
    anomaly_dataset = dataset.filter(lambda x: x['label'] != mapping[inlier_topic])
    
    n_anomalies = int((anomaly_rate * inlier_dataset.num_rows) / (1 - anomaly_rate))
    anomaly_indices = np.random.randint(0, anomaly_dataset.num_rows, n_anomalies)
    anomaly_dataset = anomaly_dataset.select(anomaly_indices)
    
    inlier_dataset = inlier_dataset.map(add_col, fn_kwargs={"anomaly_class": 0})
    anomaly_dataset = anomaly_dataset.map(add_col, fn_kwargs={"anomaly_class": 1})

    return DataLoader(inlier_dataset, batch_size=batch_size, shuffle=True), DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True)
