import os
import time
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import spacy_universal_sentence_encoder
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import arraygan.git.code.ml_model.dataconfig as config
from sklearn.metrics import precision_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from sklearn.ensemble import VotingClassifier

nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')

def MultiModel():
    for turn in range(5):
        for proecess in range(1, 16):
            df = pd.read_csv('../../../model/preprocess_data/set_{}/data.csv'.format(str(proecess)))
            df_test_1 = pd.read_csv('../../../model/preprocess_data/set_{}/df_test_1.csv'.format(str(proecess)))

            # df = pd.read_csv('../../data/n-mask/set_{}/data.csv'.format(str(proecess)))
            # df_test_1 = pd.read_csv('../../data/n-mask/set_{}/data.csv'.format(str(proecess)))

            df_claim = df['claims']
            df_class = df['classes']

            X_test_1 = df_test_1['claims']
            y_test_1 = df_test_1['classes']

            df_claim = df_claim.values.tolist()

            df_claim_encode = []
            df_evidence_encode = []

            X_test_1 = X_test_1.values.tolist()
            y_test_1 = y_test_1.values.tolist()

            for claim in df_claim:
                temp = nlp(claim)
                df_claim_encode.append(temp.vector)

            X_test_encode = []
            for test_data in X_test_1:
                temp = nlp(test_data)
                X_test_encode.append(temp.vector)

            df_claim_evidence_encode = pd.DataFrame(df_claim_encode)
            X_test_encode = pd.DataFrame(X_test_encode)

            # base_models = [
            #     ('SVM', SVC()),
            #     ('Random Forest', RandomForestClassifier()),
            #     ('XGB', GradientBoostingClassifier()),
            # ]
            #
            # stacked = StackingClassifier(
            #     estimators=base_models,
            #     final_estimator=RandomForestClassifier(),
            #     cv=5)
            #
            start_time = time.time()
            # stacked.fit(df_claim_evidence_encode, df_class)
            # stacked_prediction = stacked.predict(X_test_encode)

            clf1 = RandomForestClassifier(n_estimators=1000, random_state=42)

            clf2 = svm.SVC(C=4.0, kernel='rbf', random_state=42)

            # model = VotingClassifier(estimators=[('LR', clf1), ('RF', clf2)],
            #                          voting='soft', weights=[1, 2])
            model = VotingClassifier(estimators=[('dtc', RandomForestClassifier(n_estimators=1000, random_state=42)),
                            ('lr', svm.SVC(C=4.0, kernel='rbf', probability=True, random_state=42))],voting='soft',  weights=[1, 2])

            model.fit(df_claim_evidence_encode, df_class)
            voting_predictions = model.predict(X_test_encode)

            filename = '../model/voting_model_process_{}_turn_{}'.format(str(proecess), str(turn))
            pickle.dump(model, open(filename, 'wb'))


            end_time = time.time()

            recall = recall_score(y_test_1, voting_predictions)
            precision = precision_score(y_test_1, voting_predictions)
            accuracy = accuracy_score(y_test_1, voting_predictions)
            f1 = f1_score(y_test_1, voting_predictions)
            classification_score = classification_report(y_test_1, voting_predictions, digits=4)
            print("-------Stacked Ensemble-------")
            print("Recall: {}".format(recall))
            print("precision: {}".format(precision))
            print("F1: {}".format(f1))
            print("Computation Time: {}".format(end_time - start_time))
            print("----------------------------------")
            if (os.path.exists(config.STACK_OUT_PUT_FILE)):
                f = open(config.STACK_OUT_PUT_FILE, "a")
                f.write('For testset 1 Stack Jan 13  weights=[1, 2]')
                f.write('\n\n')
            else:
                f = open(config.STACK_OUT_PUT_FILE, "a")
            f.write('\n\n')
            f.write('\n\n')
            f.write('Turn: ' + str(turn))
            f.write('\n\n')
            f.write('\n')
            f.write('Process: ' + str(proecess))
            f.write('\n\n')
            f.write("Precision:" + str(precision))
            f.write('\n')
            f.write(" Recall: " + str(recall))
            f.write('\n')
            f.write(" Accuracy: " + str(accuracy))
            f.write('\n')
            f.write(" F1_macro: " + str(f1))
            f.write('\n')
            f.write(classification_score)
            f.close()

            print("f1 score = ", f1)
            print("precision = ", precision)
            print("recall = ", recall)
            print("accuracy_metric micro", accuracy)
            print(classification_score)
            print('Done Turn: {}, Process: {}'.format(turn, proecess))

