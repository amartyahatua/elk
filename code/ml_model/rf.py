import  pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import arraygan.git.code.ml_model.dataconfig as config
import os
import pickle

from sklearn.decomposition import IncrementalPCA
#import spacy_universal_sentence_encoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import spacy_universal_sentence_encoder

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# load one of the models: ['en_use_md', 'en_use_lg', 'xx_use_md', 'xx_use_lg']
#Universal sentence encoder
nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')

#BERT Sentence encoder
#nlp = SentenceTransformer('all-MiniLM-L6-v2')

# get two documents
# doc_1 =

# sbert_model = spacy_universal_sentence_encoder('paraphrase-distilroberta-base-v1')

def apply_pca(data):
    pca = IncrementalPCA(n_components=128)
    principalComponents = pca.fit_transform(data)
    return principalComponents


def RandomForest():
    for turn in range(5):
        for proecess in range(1, 16):
            df = pd.read_csv('../../data/l-mask/set_{}/FEVER_Train.csv'.format(str(proecess)))
            df_test_1 = pd.read_csv('../../data/l-mask/set_{}/FEVER_1_Test.csv'.format(str(proecess)))

            df_val = df

            df_claim = df['claims'].to_numpy()
            df_class = df['class'].to_numpy()

            X_test_1 = df_test_1['claims'].to_numpy()
            y_test_1 = df_test_1['classes'].to_numpy()

            print(type(df_claim))
            print(df_class.shape)
            print(X_test_1.shape)
            print(y_test_1.shape)



            #df_claim = df_claim.values.tolist()


            df_claim_encode = []
            df_evidence_encode = []


            # X_test_1 = X_test_1.values.tolist()
            # y_test_1 = y_test_1.values.tolist()



            for claim in df_claim:
                temp_all_dim = nlp(claim)
                df_claim_encode.append(temp_all_dim.vector)
            df_claim_encode = apply_pca(df_claim_encode)


            X_test_encode = []
            for test_data in X_test_1:
                temp_all_dim = nlp(test_data)
                X_test_encode.append(temp_all_dim.vector)
            X_test_encode = apply_pca(X_test_encode)


            # tfIdfVectorizer = TfidfVectorizer()
            #
            # tfidf_train_vectors = tfIdfVectorizer.fit_transform(df_claim)
            # print(tfidf_train_vectors.shape)
            # tfidf_test_vectors = tfIdfVectorizer.transform(X_test_1)
            # print(tfidf_test_vectors.shape)

            # df_claim_evidence_encode = pd.DataFrame(df_claim_encode)
            # X_test_encode = pd.DataFrame(X_test_encode)

            clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=10,
                                         min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                                         verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
            #clf = RandomForestClassifier()

            print("Model fitting started ....")

            # scores = cross_val_score(clf, df_claim_evidence_encode, df_class, cv = 5, scoring = 'f1_macro')
            # print('Scores: ',scores)

            model = clf.fit(df_claim_encode, df_class)
            print('Training done')
            filename = '../model/tfidf_random_forest_model_process_{}_turn_{}.pkl'.format(str(proecess),str(turn))
            pickle.dump(model, open(filename, 'wb'))

            predict_train = model.predict(df_claim_encode)
            f1_train = f1_score(df_class, predict_train)


            svc_predictions = model.predict(X_test_encode)

            precision_metric_micro = precision_score(y_test_1, svc_predictions)
            recall_metric_micro = recall_score(y_test_1, svc_predictions)
            accuracy_metric_micro = accuracy_score(y_test_1, svc_predictions)
            f1_metric_micro = f1_score(y_test_1, svc_predictions)
            classification_score = classification_report(y_test_1, svc_predictions, digits=4)

            if (os.path.exists(config.RF_NORMALIZED_OUT_PUT_FILE)):
                f = open(config.RF_NORMALIZED_OUT_PUT_FILE, "a")
                f.write('For Test 1 linguist enrichment using Sentence BERT 20th Feb')
            else:
                f = open(config.RF_NORMALIZED_OUT_PUT_FILE, "a")
            print(config.RF_NORMALIZED_OUT_PUT_FILE)
            f.write('\n\n')
            f.write('Turn: '+ str(turn))
            f.write('   Process: '+str(proecess))
            f.write('\n')
            f.write("precision_macro:" + str(precision_metric_micro))
            f.write('\n')
            f.write(" recall_macro: " + str(recall_metric_micro))
            f.write('\n')
            f.write(" Accuracy: " + str(accuracy_metric_micro))
            f.write('\n')
            f.write(" f1_macro: " + str(f1_metric_micro))
            f.write('\n')
            f.write(" Classification score: " + str(classification_score))
            f.write('\n')
            print('Done Turn: {}, Process: {}'.format(turn, proecess))
            f.close()

            print("f1 train", f1_train)
            print("precision_metric micro", precision_metric_micro)
            print("recall_metric micro", recall_metric_micro)
            print("accuracy_metric micro", accuracy_metric_micro)
            print("f1_metric micro", f1_metric_micro)
            print(classification_score)
            print('Done Turn: {}, Process: {}'.format(turn, proecess))

RandomForest()



