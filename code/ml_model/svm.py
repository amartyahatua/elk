from sentence_transformers import SentenceTransformer
import  pandas as pd
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import arraygan.git.code.ml_model.dataconfig as config
import os
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import classification_report


# sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
#Universal sentence encoder
#nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')

#BERT Sentence encoder
nlp = SentenceTransformer('all-MiniLM-L6-v2')


def apply_pca(data):
    pca = IncrementalPCA(n_components=128)
    principalComponents = pca.fit_transform(data)
    return principalComponents


def SVM():
    for turn in range(5):
        for proecess in range(1,15):
            df = pd.read_csv('../../../model/preprocess_data/set_{}/data.csv'.format(str(proecess)))
            df_test = pd.read_csv('../../../model/preprocess_data/set_{}/df_test_1.csv'.format(str(proecess)))

            df_claim = df['claims']
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
                temp_all_dim = nlp.encode(claim)
                df_claim_encode.append(temp_all_dim)
            df_claim_encode = apply_pca(df_claim_encode)

            X_test_encode = []
            for test_data in X_test:
                temp_all_dim = nlp.encode(test_data)
                X_test_encode.append(temp_all_dim)
            X_test_encode = apply_pca(X_test_encode)


            df_claim_evidence_encode = pd.DataFrame(df_claim_encode)
            X_test_encode = pd.DataFrame(X_test_encode)

            clf = svm.SVC(C=4.0, kernel='rbf', degree=7, gamma='scale', coef0=0.0,
                          shrinking=True, probability=False, tol=0.001, cache_size=200,
                          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                          break_ties=False, random_state=None)
            print("Model fitting started ....")
            model = clf.set_params(kernel='rbf').fit(df_claim_evidence_encode, df_class)

            filename = '../model/sent_bert_svm_model_process_{}_turn_{}.pkl'.format(str(proecess), str(turn))
            pickle.dump(model, open(filename, 'wb'))

            svc_predictions = model.predict(X_test_encode)

            precision_metric_micro = precision_score(y_test, svc_predictions)
            recall_metric_micro = recall_score(y_test, svc_predictions)
            accuracy_metric_micro = accuracy_score(y_test, svc_predictions)
            f1_metric_micro = f1_score(y_test, svc_predictions)
            classification_score = classification_report(y_test, svc_predictions, digits=4)

            if (os.path.exists(config.SVM_NORMALIZED_OUT_PUT_FILE)):
                f = open(config.SVM_NORMALIZED_OUT_PUT_FILE, "a")
                f.write('For Test 1 linguist enrichment using Sentence BERT 14 Jan')
                f.write('\n\n')
            else:
                f = open(config.SVM_NORMALIZED_OUT_PUT_FILE, "a")
            f.write('\n\n')
            f.write('\n\n')
            f.write('Turn: '+ str(turn))
            f.write('\n\n')
            f.write('\n')
            f.write('   Process: '+ str(proecess))
            f.write('\n\n')
            f.write("precision_macro:" + str(precision_metric_micro))
            f.write(" recall_macro: " + str(recall_metric_micro))
            f.write('\n')
            f.write(" Accuracy: " + str(accuracy_metric_micro))
            f.write('\n')
            f.write(" f1_macro: " + str(f1_metric_micro))
            f.write('\n')
            f.write("Classification score: " + str(classification_score))
            f.write('\n')
            f.close()

            print("precision_metric micro", precision_metric_micro)
            print("recall_metric micro", recall_metric_micro)
            print("accuracy_metric micro", accuracy_metric_micro)
            print("f1_metric micro", f1_metric_micro)
            print(classification_score)
            print('Done Turn: {}, Process: {}'.format(turn, proecess))
