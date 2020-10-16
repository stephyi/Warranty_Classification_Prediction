
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE


def DealwithSample(data, label, method="ADA"):
    if method== "ADA":
        ada = ADASYN(random_state=42)
        X_res, y_res = ada.fit_resample(data, label)
        return X_res, y_res 
    elif method == "RandomOverSampler" :
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(data, label)
        return X_res, y_res
    elif method == "SMOTE" :
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(data, label)
        return X_res, y_res


