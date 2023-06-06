import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from sklearn import svm
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


def SVM():
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
                tfIdfVectorizer = TfidfVectorizer(lowercase=False)

                tfidf_train_vectors = tfIdfVectorizer.fit_transform(df_claims)
                tfidf_test_vectors_1 = tfIdfVectorizer.transform(df_test_1_claims)
                tfidf_test_vectors_2 = tfIdfVectorizer.transform(df_test_2_claims)

                clf = svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0,
                          shrinking=True, probability=False, tol=0.001, cache_size=200,
                          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                          break_ties=False, random_state=None)

                model = clf.fit(tfidf_train_vectors, df_classes)
                filename = '../model/TFIDF_SVM_Cleaning_model_process_{}_turn_{}_GT.pkl'.format(str(set + 1), str(process + 1))
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
                tfIdfVectorizer = TfidfVectorizer(lowercase=False)

                tfidf_train_vectors = tfIdfVectorizer.fit_transform(df_claims_clean)
                tfidf_test_vectors_1 = tfIdfVectorizer.transform(df_test_1_claims_clean)
                tfidf_test_vectors_2 = tfIdfVectorizer.transform(df_test_2_claims_clean)

                clf = svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0,
                          shrinking=True, probability=False, tol=0.001, cache_size=200,
                          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                          break_ties=False, random_state=None)
                model = clf.fit(tfidf_train_vectors, df_classes)
                predicted_fever_1 = model.predict(tfidf_test_vectors_1)
                predicted_fever_2 = model.predict(tfidf_test_vectors_2)

                filename = '../model/TFIDF_SVM_Cleaning_model_process_{}_turn_{}_Cleaning.pkl'.format(str(set + 1),
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
                tfIdfVectorizer = TfidfVectorizer(lowercase=False)

                tfidf_train_vectors = tfIdfVectorizer.fit_transform(df_claims_clean_no_space)
                tfidf_test_vectors_1 = tfIdfVectorizer.transform(df_test_1_claims_clean_no_space)
                tfidf_test_vectors_2 = tfIdfVectorizer.transform(df_test_2_claims_clean_no_space)

                clf = svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0,
                          shrinking=True, probability=False, tol=0.001, cache_size=200,
                          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                          break_ties=False, random_state=None)
                model = clf.fit(tfidf_train_vectors, df_classes)
                predicted_fever_1 = model.predict(tfidf_test_vectors_1)
                predicted_fever_2 = model.predict(tfidf_test_vectors_2)

                filename = '../model/TFIDF_SVM_Cleaning_model_process_{}_turn_{}_Clean_No_Space.pkl'.format(str(set + 1),
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
                tfIdfVectorizer = TfidfVectorizer(lowercase=False)

                tfidf_train_vectors = tfIdfVectorizer.fit_transform(df_claims_clean_space)
                tfidf_test_vectors_1 = tfIdfVectorizer.transform(df_test_1_claims_clean_space)
                tfidf_test_vectors_2 = tfIdfVectorizer.transform(df_test_2_claims_clean_space)

                clf = svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0,
                          shrinking=True, probability=False, tol=0.001, cache_size=200,
                          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                          break_ties=False, random_state=None)

                model = clf.fit(tfidf_train_vectors, df_classes)
                predicted_fever_1 = model.predict(tfidf_test_vectors_1)
                predicted_fever_2 = model.predict(tfidf_test_vectors_2)
                filename = '../model/TFIDF_SVM_Cleaning_model_process_{}_turn_{}_Clean_Space.pkl'.format(str(set + 1),
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
    final_result.to_csv('../../data/SVM_TFIDF_Cleaning_Result.csv', index=False)


SVM()
