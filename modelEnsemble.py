import pandas as pd

from sklearn.preprocessing import LabelEncoder

# for the models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# for evaluation metrics of the models
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score


class modelEnsemble:
    def __init__(self, models, x_train, y_train, x_test, y_test):
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.sc = None

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
                                    "Accuracy", "Precision", "Recall", "F1-Measure"], index=['Voting-Hard', 'Voting-Soft'])
        return votingReport

    def make_report(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="macro")
        recall = recall_score(self.y_test, y_pred, average="macro")
        f1 = f1_score(self.y_test, y_pred, average="macro")
        metrics = [accuracy, precision, recall, f1]
        return metrics

    def stacking(self, finalmodel):
        sc = StackingClassifier(estimators=self.models,
                                final_estimator=finalmodel)
        sc = sc.fit(self.x_train, self.y_train)
        sc_pred = sc.predict(self.x_test)
        sc_prob = sc.predict_proba(self.x_test)
        scMetric = self.make_report(sc_pred, sc_prob)
        self.sc = sc

        return pd.DataFrame(scMetric, columns=['Stacking'], index=["Accuracy", "Precision", "Recall", "F1-Measure", "AUC"])

    def predict(self, x):
        return self.sc.predict(x)


class modelReport:
    def __init__(self, y_test):
        self.y_test = y_test
        self.metrics = []
        self.names = []

    def addmodel(self, y_prob, name, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="macro")
        recall = recall_score(self.y_test, y_pred, average="macro")
        f1 = f1_score(self.y_test, y_pred, average="macro")
        auc = roc_auc_score(self.y_test, y_prob, multi_class="ovo")
        metric = [accuracy, precision, recall, f1, auc]
        self.metrics.append(metric)
        self.names.append(name)

    def makeReport(self):
        metrics_summary = pd.DataFrame(self.metrics, columns=[
                                       "Accuracy", "Precision", "Recall", "F1-Measure", "AUC"], index=self.names)
        return metrics_summary


class OnehotReportGenerator:
    def __init__(self, traindf, testdf):
        train_data = pd.read_csv(traindf)
        self.X_train = train_data.drop(["category"], axis="columns")
        y_train = train_data[['category']]

        test_data = pd.read_csv(testdf)
        self.X_test = test_data.drop(["category"], axis="columns")
        y_test = test_data[['category']]

        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.y_test_encoded = self.label_encoder.transform(y_test)
        self.report = modelReport(self.y_test_encoded)

    def runBasicModels(self, models):
        for name, model in models:
            model.fit(self.X_train, self.y_train_encoded)
            pred = model.predict(self.X_test)
            pred_prob = model.predict_proba(self.X_test)
            self.report.addmodel(pred_prob, name, pred)
            print("name: " + name)
            print(classification_report(self.y_test_encoded,
                  pred, target_names=self.label_encoder.classes_))

    def makeReport(self):
        return self.report.makeReport()
