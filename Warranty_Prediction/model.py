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


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_recall_curve, f1_score
from sklearn.metrics import classification_report
# "Support vector classifier"
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class Modelling():
    def __init__(self):
        pass

    def _splitdata(self, data, label, size =0.1):
        x_train, x_test, y_train, y_test = train_test_split(
            data, label, test_size=size, random_state=50)
        return x_train, x_test, y_train, y_test

    def Prediction(self, x_train, x_test, y_train, y_test, method="Logistic"):
        if method== "Logistic":
            model = LogisticRegression(random_state=6, max_iter=600)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_predpr = model.predict_proba(x_test)
            model.score(x_train, y_train)
            print(confusion_matrix(y_test, y_pred))
            print(log_loss(y_test, y_predpr))
            print(classification_report(y_test, y_pred))
        elif method == "SupportVector":
            model = SVC(kernel='rbf',verbose=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
        elif method == "RandomForest":
            model = GridSearchCV(RandomForestClassifier(), param_grid={"n_estimators":[100,200], "max_depth" : [20, 30, 40, 50, 60, 70], "min_samples_leaf":[100, 200, 300, 400]}, verbose=1, cv=5)
            model.fit(x_train, y_train)
            print(model.best_params_)
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_pred, y_test))
            print(classification_report(y_test, y_pred))
            print(roc_auc_score(y_test, y_pred))
            return model

        elif method == "Decisiontree":
            model = DecisionTreeClassifier()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_pred, y_test))
            print(classification_report(y_test, y_pred))
        elif method == "multilayerperceptron":
            model = MLPClassifier(early_stopping=True, verbose=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_pred, y_test))
            print(classification_report(y_test, y_pred))
            print(roc_auc_score(y_test, y_pred))

        elif method == "gradientboosting":
            model = GradientBoostingClassifier(max_depth=200, verbose=1,min_samples_leaf=30 )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print(confusion_matrix(y_pred, y_test))
            print(classification_report(y_test, y_pred))
            print(roc_auc_score(y_test, y_pred))