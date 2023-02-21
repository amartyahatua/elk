from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import  pandas as pd
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import arraygan.git.code.ml_model.dataconfig as config
import os
import spacy_universal_sentence_encoder
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier

# sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')

def Voting():
    for turn in range(5):
        for proecess in range(1,2):
            df = pd.read_csv('../../data/data.csv'.format(str(proecess)))
            df_test = pd.read_csv('../../data/FEVER_1_Test.csv'.format(str(proecess)))
            df_val = df

            df_claim = df['claims']
            # df_evidence = df['evidence']
            df_class = df['classes']

            df_claim = df_claim.values.tolist()
            df_class = df_class.values.tolist()

            X_test = df_test['claims']
            y_test = df_test['classes']

            df_claim_encode = []
            df_evidence_encode = []

            X_test = X_test.values.tolist()
            y_test = y_test.values.tolist()

            for claim in df_claim:
                temp = nlp(claim)
                df_claim_encode.append(temp.vector)

            X_test_encode = []
            for test_data in X_test:
                temp = nlp(test_data)
                X_test_encode.append(temp.vector)

            claim_evidence = []
            for i in range(len(df_claim)):
                clm = df_claim[i]
                # eve = df_evidence[i]
                temp = clm
                claim_evidence.append(temp)
            df_claim_evidence_encode = []
            for evidence in claim_evidence:
                temp = nlp(evidence)
                df_claim_evidence_encode.append(temp.vector)

            df_claim_evidence_encode = pd.DataFrame(df_claim_evidence_encode)

            # create a voting classifier with hard voting
            voting_classifier_hard = VotingClassifier(
                estimators=[('dtc', RandomForestClassifier(n_estimators=10000, criterion='gini', max_depth=None,
                                                           min_samples_split=10,
                                                           min_samples_leaf=10, min_weight_fraction_leaf=0.0,
                                                           max_features='sqrt', max_leaf_nodes=None,
                                                           min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                                           n_jobs=None, random_state=None,
                                                           verbose=0, warm_start=False, class_weight=None,
                                                           ccp_alpha=0.0, max_samples=None)),
                            ('lr', svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0,
                                           shrinking=True, probability=True, tol=0.001, cache_size=200,
                                           class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                                           break_ties=False, random_state=None)),
                            ('gnb', GradientBoostingClassifier(learning_rate=0.01,
                                                               n_estimators=10000, subsample=1.0,
                                                               criterion='friedman_mse',
                                                               min_samples_split=2, min_samples_leaf=1,
                                                               min_weight_fraction_leaf=0.0, max_depth=3,
                                                               min_impurity_decrease=0.0, init=None, random_state=None,
                                                               max_features=None, verbose=0, max_leaf_nodes=None,
                                                               warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
                                                               tol=0.0001, ccp_alpha=0.0))],
                voting='hard')

            # create a voting classifier with soft voting
            voting_classifier_soft = VotingClassifier(
                estimators=[('dtc', RandomForestClassifier(n_estimators=10000, criterion='gini', max_depth=None, min_samples_split=10,
                                         min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                                         verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)),
                            ('lr', svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0,
                          shrinking=True, probability=True, tol=0.001, cache_size=200,
                          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                          break_ties=False, random_state=None)),
                            ('gnb', GradientBoostingClassifier(learning_rate=0.01,
                                                               n_estimators=10000, subsample=1.0,
                                                               criterion='friedman_mse',
                                                               min_samples_split=2, min_samples_leaf=1,
                                                               min_weight_fraction_leaf=0.0, max_depth=3,
                                                               min_impurity_decrease=0.0, init=None, random_state=None,
                                                               max_features=None, verbose=0, max_leaf_nodes=None,
                                                               warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
                                                               tol=0.0001, ccp_alpha=0.0))],
                voting='soft')

            # make predictions with the hard voting ml_model
            voting_classifier_hard.fit(df_claim_evidence_encode, df_class)
            y_pred_vch = voting_classifier_hard.predict(X_test_encode)

            # make predictions with the soft voting ml_model
            voting_classifier_soft.fit(df_claim_evidence_encode, df_class)
            y_pred_vcs = voting_classifier_soft.predict(X_test_encode)

            # evaluate both models with the f-1 score
            # f1_vch = f1_score(y_test, y_pred_vch)
            # f1_vcs = f1_score(y_test, y_pred_vcs)

            precision_metric_micro_h = precision_score(y_test, y_pred_vch)
            recall_metric_micro_h = recall_score(y_test, y_pred_vch)
            accuracy_metric_micro_h = accuracy_score(y_test, y_pred_vch)
            f1_metric_micro_h = f1_score(y_test, y_pred_vch)

            precision_metric_micro_s = precision_score(y_test, y_pred_vcs)
            recall_metric_micro_s = recall_score(y_test, y_pred_vcs)
            accuracy_metric_micro_s = accuracy_score(y_test, y_pred_vcs)
            f1_metric_micro_s = f1_score(y_test, y_pred_vcs)

            if (os.path.exists(config.VOTING_NORMALIZED_OUT_PUT_FILE)):
                f = open(config.VOTING_NORMALIZED_OUT_PUT_FILE, "a")
                f.write('For testset 1  Normalized 7th Nov')
                f.write('\n\n')
            else:
                f = open(config.VOTING_NORMALIZED_OUT_PUT_FILE, "a")
            f.write('\n\n')
            f.write('\n\n')
            f.write('Turn: '+ str(turn))
            f.write('\n\n')
            f.write('\n')
            f.write('   Process: '+ str(proecess))
            f.write('\n\n')
            f.write("precision_macro hard:" + str(precision_metric_micro_h))
            f.write(" recall_macro hard: " + str(recall_metric_micro_h))
            f.write('\n')
            f.write(" Accuracy hard: " + str(accuracy_metric_micro_h))
            f.write('\n')
            f.write(" f1_macro hard: " + str(f1_metric_micro_h))
            f.write('\n')
            f.write("precision_macro soft:" + str(precision_metric_micro_s))
            f.write(" recall_macro soft: " + str(recall_metric_micro_s))
            f.write('\n')
            f.write(" Accuracy soft: " + str(accuracy_metric_micro_s))
            f.write('\n')
            f.write(" f1_macro soft: " + str(f1_metric_micro_s))
            f.write('\n')
            f.close()

            print("precision_metric micro hard", precision_metric_micro_h)
            print("recall_metric micro hard", recall_metric_micro_h)
            print("accuracy_metric micro hard", accuracy_metric_micro_h)
            print("f1_metric micro hard", f1_metric_micro_h)

            print("precision_metric micro soft", precision_metric_micro_s)
            print("recall_metric micro soft", recall_metric_micro_s)
            print("accuracy_metric micro soft", accuracy_metric_micro_s)
            print("f1_metric micro soft", f1_metric_micro_s)

            print('Done Turn: {}, Process: {}'.format(turn, proecess))

Voting()