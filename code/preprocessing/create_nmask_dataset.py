import pandas as pd
from arraygan.git.code.preprocessing import preprocess_nmask
import os
import json

#ner_normalized
#ner = NER()

with open('../../data/ner_normalized.json', 'r') as json_file:
    ner_normalized = json.load(json_file)

for method in range(1,13):
    if(method == 1):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_1_1(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_1_1(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_1_1(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 1')

    elif(method == 2):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_1_2(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_1_2(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_1_2(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 2')

    elif(method == 3):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_1_3(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_1_3(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_1_3(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 3')

    elif(method == 4):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_1_4(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_1_4(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_1_4(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 4')

    elif(method == 5):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_2_1(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_2_1(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_2_1(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 5')

    elif(method == 6):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_2_2(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_2_2(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_2_2(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 6')

    elif(method == 7):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_2_3(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_2_3(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_2_3(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 7')

    elif(method == 8):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_2_4(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_2_4(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_2_4(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 8')

    elif(method == 9):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_3_1(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_3_1(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_3_1(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 9')

    elif(method == 10):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_3_2(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_3_2(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_3_2(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 10')

    elif(method == 11):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_3_3(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_3_3(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_3_3(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 11')

    elif(method == 12):
        df = pd.read_csv('../../data/data.csv')
        df_claim_preprocess = preprocess_nmask.preprocess_method_3_4(df['claims'], ner_normalized)
        df['claims'] = df_claim_preprocess

        df_test_1 = pd.read_csv('../../data/FEVER_1_Test.csv')
        df_test_1_preprocess = preprocess_nmask.preprocess_method_3_4(df_test_1['claims'], ner_normalized)
        df_test_1['claims'] = df_test_1_preprocess

        df_test_2 = pd.read_csv('../../data/FEVER_2_Test.csv')
        df_test_2_preprocess = preprocess_nmask.preprocess_method_3_4(df_test_2['claims'], ner_normalized)
        df_test_2['claims'] = df_test_2_preprocess
        print('Done 12')


    out_dir = '../data/n-mask/set_{}'.format(str(method))
    df_file = '../data/n-mask/set_{}/data_nmask_person.csv'.format(str(method))
    out_file_1 = '../data/n-mask/set_{}/FEVER_1_Test_nmask_person.csv'.format(str(method))
    out_file_2 = '../data/n-mask/set_{}/FEVER_2_Test_nmask_person.csv'.format(str(method))
    isExist = os.path.exists(out_dir)
    if not isExist:
        os.makedirs(out_dir)
        print("The new directory is created!")
    df.to_csv(df_file, index=False)
    df_test_1.to_csv(out_file_1, index=False)
    df_test_2.to_csv(out_file_2, index=False)