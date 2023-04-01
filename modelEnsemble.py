from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
import pandas as pd


class modelEnsemble:
    def __init__(self, models, x_train, y_train, x_test, y_test):
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def voting(self):
        vcHard = VotingClassifier(estimators=self.models, voting='hard')
        vcHard = vcHard.fit(self.x_train, self.y_train)
        vcHard_pred = vcHard.predict(self.x_test)
        vcHardMetric = self.make_report(vcHard_pred)

        vcSoft = VotingClassifier(estimators=self.models, voting='soft')
        vcSoft = vcSoft.fit(self.x_train, self.y_train)
        vcSoft_pred = vcSoft.predict(self.x_test)
        vcSoftMetric = self.make_report(vcSoft_pred)
        vcReport = [vcHardMetric, vcSoftMetric]

        votingReport = pd.DataFrame(vcReport, columns=[
                                    "Accuracy", "Precision", "F1-Measure"], index=['Voting-Hard', 'Voting-Soft'])
        return votingReport

    def make_report(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="macro")
        # recall = recall_score(self.y_test, y_pred, average="macro")
        f1 = f1_score(self.y_test, y_pred, average="macro")
        metrics = [accuracy, precision, f1]
        return metrics

    def stacking(self, finalmodel):
        sc = StackingClassifier(estimators=self.models,
                                final_estimator=finalmodel)
        sc = sc.fit(self.x_train, self.y_train)
        sc_pred = sc.predict(self.x_test)
        scMetric = self.make_report(sc_pred)

        return pd.DataFrame(scMetric, columns=['Stacking'], index=["Accuracy", "Precision", "F1-Measure"])

class modelReport:
    def __init__(self, y_test):
        self.y_test = y_test
        self.metrics = []
        self.names = []
    
    def addmodel(self, y_prob, name):
        accuracy = accuracy_score(self.y_test, y_prob)
        precision = precision_score(self.y_test, y_prob, average = "macro")
        recall = recall_score(self.y_test, y_prob, average = "macro")
        f1 = f1_score(self.y_test, y_prob, average = "macro")
        auc = roc_auc_score(self.y_test, y_prob, multi_class = "ovo")
        metric = [accuracy, precision, recall, f1, auc]
        self.metrics.append(metric)
        self.names.append(name)

    def makeReport(self):
        metrics_summary = pd.DataFrame(self.metrics, columns = ["Accuracy", "Precision", "Recall", "F1-Measure", "AUC"], index = self.names)
        return metrics_summary
