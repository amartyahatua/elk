#path:
POSITIVE_DATAPATH = '../../output/data_small.txt'
NEGATIVE_DATAPATH = '../../output/data_small.txt'
GROUNDTRUTH_PATH = '../../output/data_small.csv'
TESTDATA1_PATH = '../../output/data_small.csv'
TESTDATA2_PATH = '../../output/data_small.csv'
OUT_PUT_FILE = '../../output/result.txt'
OUT_PUT_FILE_LSTM = '../../output/result_lstm.txt'
SVM_OUT_PUT_FILE = '../../output/result_svm.txt'
RF_OUT_PUT_FILE = '../../output/result_rf.txt'
SDG_OUT_PUT_FILE = '../../output/result_sdg.txt'
BERT_OUT_PUT_FILE = '../../output/result_bert.txt'
XGB_OUT_PUT_FILE = '../../output/result_xgb.txt'
STACK_OUT_PUT_FILE = '../../output/old/result_stack.txt'
VOTING_NORMALIZED_OUT_PUT_FILE = '../../output/result_normalized_voting.txt'


XGB_NORMALIZED_OUT_PUT_FILE = '../../output/old/result_normalized_xgb.txt'
SVM_NORMALIZED_OUT_PUT_FILE = '../../output/old/result_normalized_svm.txt'
RF_NORMALIZED_OUT_PUT_FILE = '../../output/old/result_normalized_rf.txt'
STACK_NORMALIZED_OUT_PUT_FILE = '../../output/result_normalized_stack.txt'

#feature_config:
FEATURE_DIMENSION = 10

#train_config:
BATCH_SIZE = 100
HIDDEN_LAYER = 32
LEAK = 0.2
MAP_DIM = 16
SINGLE_LAYER = 1
EPOCHS = 200
BETA1 = 0.05
BETA2 = 0.005
SAMPLE_SIZE = 100
SAMPLE_STEP = 50
SOFT_LABEL = 0.5
LR = 0.001
