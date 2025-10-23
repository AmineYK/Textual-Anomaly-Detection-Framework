from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

import evaluation as ev

def ocsvm_pipeline(X, y, kernel='rbf', gamma='scale', nu=0.1, test_size=0.2, verbose=True, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # train only on intlier class
    X_train = X_train[y_train == 0]   

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    ocsvm.fit(X_train_scaled)

    y_test_pred = ocsvm.predict(X_test_scaled)
    y_test_pred = np.where(y_test_pred == 1, 0, 1)  
    scores_test = -ocsvm.decision_function(X_test_scaled) 
    
    if verbose:
        
        print("=== One-Class SVM Baseline ===")
        
        print("\nTesting Results")
        ocsvm_auc_test, ocsvm_f1_test, ocsvm_precision_test, ocsvm_recall_test = ev.evaluation(y_test, scores_test, y_test_pred, verbose=True)
    
    
    return  ocsvm,(ocsvm_auc_test, ocsvm_f1_test, ocsvm_precision_test, ocsvm_recall_test)
            