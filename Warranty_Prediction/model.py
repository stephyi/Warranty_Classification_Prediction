from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
#from keras.models import Sequential
#from keras.layers import Activation, Dense, Dropout, BatchNormalization
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_recall_curve, f1_score
from sklearn.metrics import classification_report
# "Support vector classifier"
from sklearn.svm import SVC
from sklearn.neural_network import multilayer_perceptron


class Modelling():
    def __init__(self, data,  label ):
        self.data = data
        self.label = label 

    
    def _splitdata(self, data, label):
        x_train, x_test, y_train, y_test = train_test_split(
            data, label, stratify=label, test_size=.10, random_state=50)
        return x_train, x_test, y_train, y_test

    def Prediction(self, x_train, x_test, y_train, y_test, method="Logistic"):
        if method== "Logistic":
            model = LogisticRegression(random_state=6)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_predpr = model.predict_proba(x_test)
            model.score(x_train, y_train)
            print(confusion_matrix(y_test, y_pred))
            print(log_loss(y_test, y_predpr))
            print(classification_report(y_test, y_pred))
        elif method == "SupportVector":
            model = SVC(kernel='linear', C=1E10)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_predpr = model.predict_proba(x_test)
            model.score(x_train, y_train)
            print(confusion_matrix(y_test, y_pred))
            print(log_loss(y_test, y_predpr))
            print(classification_report(y_test, y_pred))
        elif method == "RandomForest":
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_pred, y_test))
            print(classification_report(y_test, y_pred))
        elif method == "Decisiontree":
            model = DecisionTreeClassifier()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_pred, y_test))
            print(classification_report(y_test, y_pred))

    






