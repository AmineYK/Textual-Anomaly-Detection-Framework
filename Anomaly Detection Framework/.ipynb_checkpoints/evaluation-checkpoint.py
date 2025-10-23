import numpy as np 
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


# Cosinus Similarity
#--------------------------------

def cosinus_similarity(emb1,emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def evaluation(y_true, scores, y_pred, verbose=True):
    
    auc = roc_auc_score(y_true, scores)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if verbose:
        print(f"AUC:        {auc:.4f}")
        print(f"F1-score:   {f1:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")

    return auc, f1, precision, recall
