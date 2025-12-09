from Modelisation.Baselines.baseline import BaselineModel
import Modelisation.evaluation as ev
from pyod.models.auto_encoder import AutoEncoder

class AE(BaselineModel):

    def __init__(self, args):
        self.model = AutoEncoder(**args)

    def train(self, X_train):
        
        self.model.fit(X_train.cpu())

        return self.model

    def test(self, X_test, y_test):        

        scores = self.model.decision_function(X_test.cpu())

        auc_ae, fpr95_ae, ap_ae = ev.evaluation(y_test, scores, verbose=False)

        return auc_ae, fpr95_ae, ap_ae 
