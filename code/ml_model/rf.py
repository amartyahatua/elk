import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# load one of the models: ['en_use_md', 'en_use_lg', 'xx_use_md', 'xx_use_lg']
# Universal sentence encoder
# nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')

# BERT Sentence encoder
# nlp = SentenceTransformer('all-MiniLM-L6-v2')

# get two documents
# doc_1 =

# sbert_model = spacy_universal_sentence_encoder('paraphrase-distilroberta-base-v1')

def apply_pca(data):
    pca = IncrementalPCA(n_components=128)
    principalComponents = pca.fit_transform(data)
    return principalComponents


sets = 5
processes = 3

data_path_clean = '../../data/clean'
data_path_preprocess = '../../data/preprocessing'

col_name = [['Set', 'Process', \
             'GT Precision', 'GT Recall', 'GT F1', \
             'Clean Precision', 'Clean Recall', 'Clean F1', \
             'Clean No Sp Precision', 'Clean No Sp Recall', 'Clean No Sp F1', \
             'Clean Sp Precision', 'Clean Sp Recall', 'Clean Sp F1']]


def RandomForest():
    final_result = pd.DataFrame()
    for turn in range(1):
        for set in range(sets):
            for process in range(processes):
                result = []
                result.append(process + 1)
                result.append(set + 1)
                df_train = pd.read_csv(
                    os.path.join(data_path_clean, 'fever_train', 'FEVER_Train_{}_{}.csv'.format(set + 1, process + 1)))
                df_test_1 = pd.read_csv(os.path.join(data_path_clean, 'fever_1_test',
                                                     'FEVER_1_Test_{}_{}.csv'.format(set + 1, process + 1)))
                df_test_2 = pd.read_csv(os.path.join(data_path_clean, 'fever_2_test',
                                                     'FEVER_2_Test_{}_{}.csv'.format(set + 1, process + 1)))

                df_claims = df_train['claims']
                df_claims_clean = df_train['claims_clean']
                df_claims_clean_no_space = df_train['cleaning_no_space']
                df_claims_clean_space = df_train['cleaning_space']
                df_classes = df_train['class']

                df_test_1_claims = df_test_1['claims']
                df_test_1_claims_clean = df_test_1['claims_clean']
                df_test_1_claims_clean_no_space = df_test_1['cleaning_no_space']
                df_test_1_claims_clean_space = df_test_1['cleaning_space']
                df_test_1_classes = df_test_1['classes']

                df_test_2_claims = df_test_2['claims']
                df_test_2_claims_clean = df_test_2['claims_clean']
                df_test_2_claims_clean_no_space = df_test_2['cleaning_no_space']
                df_test_2_claims_clean_space = df_test_2['cleaning_space']
                df_test_2_classes = df_test_2['classes']

                # Ground truth
                countVectorizer = CountVectorizer(lowercase=False)

                tfidf_train_vectors = countVectorizer.fit_transform(df_claims)
                tfidf_test_vectors_1 = countVectorizer.transform(df_test_1_claims)
                tfidf_test_vectors_2 = countVectorizer.transform(df_test_2_claims)

                clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=10,
                                             min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                             max_leaf_nodes=None,
                                             min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                                             random_state=None,
                                             verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                             max_samples=None)

                model = clf.fit(tfidf_train_vectors, df_classes)
                filename = '../model/TFIDF_RF_Cleaning_model_process_{}_turn_{}_GT.pkl'.format(str(set + 1), str(process + 1))
                pickle.dump(model, open(filename, 'wb'))
                predicted_fever_1 = model.predict(tfidf_test_vectors_1)
                predicted_fever_2 = model.predict(tfidf_test_vectors_2)

                classification_score_test_1 = classification_report(df_test_1_classes, predicted_fever_1, digits=4)
                precision_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][0]
                recall_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][1]
                f1_score_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_1)
                result.append(recall_test_1)
                result.append(f1_score_test_1)

                classification_score_test_2 = classification_report(df_test_2_classes, predicted_fever_2, digits=4)
                precision_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][0]
                recall_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][1]
                f1_score_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_2)
                result.append(recall_test_2)
                result.append(f1_score_test_2)

                # Clean data
                countVectorizer = CountVectorizer(lowercase=False)

                tfidf_train_vectors = countVectorizer.fit_transform(df_claims)
                tfidf_test_vectors_1 = countVectorizer.transform(df_test_1_claims)
                tfidf_test_vectors_2 = countVectorizer.transform(df_test_2_claims)

                clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=10,
                                             min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                             max_leaf_nodes=None,
                                             min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                                             random_state=None,
                                             verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                             max_samples=None)
                model = clf.fit(tfidf_train_vectors, df_classes)
                predicted_fever_1 = model.predict(tfidf_test_vectors_1)
                predicted_fever_2 = model.predict(tfidf_test_vectors_2)

                filename = '../model/TFIDF_RF_Cleaning_model_process_{}_turn_{}_Cleaning.pkl'.format(str(set + 1),
                                                                                            str(process + 1))
                pickle.dump(model, open(filename, 'wb'))

                classification_score_test_1 = classification_report(df_test_1_classes, predicted_fever_1, digits=4)
                precision_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][0]
                recall_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][1]
                f1_score_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_1)
                result.append(recall_test_1)
                result.append(f1_score_test_1)

                classification_score_test_2 = classification_report(df_test_2_classes, predicted_fever_2, digits=4)
                precision_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][0]
                recall_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][1]
                f1_score_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_2)
                result.append(recall_test_2)
                result.append(f1_score_test_2)

                # Clean no space data
                countVectorizer = CountVectorizer(lowercase=False)

                tfidf_train_vectors = countVectorizer.fit_transform(df_claims)
                tfidf_test_vectors_1 = countVectorizer.transform(df_test_1_claims)
                tfidf_test_vectors_2 = countVectorizer.transform(df_test_2_claims)

                clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=10,
                                             min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                             max_leaf_nodes=None,
                                             min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                                             random_state=None,
                                             verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                             max_samples=None)
                model = clf.fit(tfidf_train_vectors, df_classes)
                predicted_fever_1 = model.predict(tfidf_test_vectors_1)
                predicted_fever_2 = model.predict(tfidf_test_vectors_2)

                filename = '../model/TFIDF_RF_Cleaning_model_process_{}_turn_{}_Clean_No_Space.pkl'.format(str(set + 1),
                                                                                                  str(process + 1))
                pickle.dump(model, open(filename, 'wb'))

                classification_score_test_1 = classification_report(df_test_1_classes, predicted_fever_1, digits=4)
                precision_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][0]
                recall_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][1]
                f1_score_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_1)
                result.append(recall_test_1)
                result.append(f1_score_test_1)

                classification_score_test_2 = classification_report(df_test_2_classes, predicted_fever_2, digits=4)
                precision_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][0]
                recall_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][1]
                f1_score_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_2)
                result.append(recall_test_2)
                result.append(f1_score_test_2)

                # Clean with space data
                countVectorizer = CountVectorizer(lowercase=False)

                tfidf_train_vectors = countVectorizer.fit_transform(df_claims)
                tfidf_test_vectors_1 = countVectorizer.transform(df_test_1_claims)
                tfidf_test_vectors_2 = countVectorizer.transform(df_test_2_claims)

                clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=10,
                                             min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                             max_leaf_nodes=None,
                                             min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                                             random_state=None,
                                             verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                             max_samples=None)

                model = clf.fit(tfidf_train_vectors, df_classes)
                predicted_fever_1 = model.predict(tfidf_test_vectors_1)
                predicted_fever_2 = model.predict(tfidf_test_vectors_2)
                filename = '../model/TFIDF_RF_Cleaning_model_process_{}_turn_{}_Clean_Space.pkl'.format(str(set + 1),
                                                                                               str(process + 1))
                pickle.dump(model, open(filename, 'wb'))
                classification_score_test_1 = classification_report(df_test_1_classes, predicted_fever_1, digits=4)
                precision_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][0]
                recall_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][1]
                f1_score_test_1 = classification_score_test_1.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_1)
                result.append(recall_test_1)
                result.append(f1_score_test_1)

                classification_score_test_2 = classification_report(df_test_2_classes, predicted_fever_2, digits=4)
                precision_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][0]
                recall_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][1]
                f1_score_test_2 = classification_score_test_2.split('\n')[-2].split()[2:-1][2]

                result.append(precision_test_2)
                result.append(recall_test_2)
                result.append(f1_score_test_2)
                result = pd.DataFrame([result])
                final_result = pd.concat([final_result, result], axis=0)
                print('Set: {} and Process: {} done'.format(set + 1, process + 1))
    final_result.to_csv('../../data/RF_BoW_Cleaning_Result.csv', index=False)


RandomForest()
