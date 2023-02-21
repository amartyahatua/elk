import pandas as pd
from arraygan.git.code.preprocessing import preprocess
import os

for method in range(14,16):
    if(method == 1):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_1_1(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_1_1(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_1_1(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 1')

    elif(method == 2):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_1_2(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_1_2(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_1_2(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 2')

    elif(method == 3):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_1_3(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_1_3(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_1_3(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 3')

    elif(method == 4):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_2_1(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_2_1(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_2_1(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 4')

    elif(method == 5):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_2_2(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_2_2(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_2_2(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 5')

    elif(method == 6):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_2_3(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_2_3(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_2_3(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 6')

    elif(method == 7):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_3_1(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_3_1(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_3_1(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 7')

    elif(method == 8):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_3_2(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_3_2(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_3_2(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 8')

    elif(method == 9):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_3_3(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_3_3(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_3_3(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 9')

    elif(method == 10):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_4_1(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_4_1(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_4_1(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 10')

    elif(method == 11):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_4_2(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_4_2(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_4_2(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 11')

    elif(method == 12):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_4_3(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_4_3(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_4_3(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 12')

    elif(method == 13):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_5_1(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_5_1(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_5_1(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 13')

    elif (method == 14):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_5_2(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_5_2(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_5_2(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 14')

    elif (method == 15):
        df = pd.read_csv('../../data/FEVER_Train.csv')
        df_claim_preprocess = preprocess.preprocess_method_5_3(df['claims'])
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess.preprocess_method_5_3(df_test_1['claims'])
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess.preprocess_method_5_3(df_test_2['claims'])
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 15')

    out_dir = '../../data/l-mask_new/set_{}'.format(str(method))
    df_file = '../../data/l-mask_new/set_{}/FEVER_Train.csv'.format(str(method))
    out_file_1 = '../../data/l-mask_new/set_{}/FEVER_1_Test.csv'.format(str(method))
    out_file_2 = '../../data/l-mask_new/set_{}/FEVER_2_Test.csv'.format(str(method))
    isExist = os.path.exists(out_dir)
    if not isExist:
        os.makedirs(out_dir)
        print("The new directory is created!")
    df.to_csv(df_file, index=False)
    df_test_1.to_csv(out_file_1, index=False)
    df_test_2.to_csv(out_file_2, index=False)






