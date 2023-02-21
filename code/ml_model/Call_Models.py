from arraygan.git.code.ml_model.rf import RandomForest
from arraygan.git.code.ml_model.svm import SVM
from arraygan.git.code.ml_model.xgb import XGB
from arraygan.git.code.ml_model.multi_model import MultiModel

print("Calling RF")
RandomForest()
print("Calling SVM")
SVM()
print("Calling XGB")
XGB()
# print("Calling Multimodel")
MultiModel()