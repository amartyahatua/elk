import stanza
import pandas as pd
import re
import json
stanza.download('en')
nlp = stanza.Pipeline('en')

def get_last_word(word):
    return word.split('-')[-1]

def remove_special_char(df):
    clean_data = []

    for x in df:
        new_text = re.sub('<.*?>', '', x)  # remove HTML tags
        #new_text = re.sub(r'[^\w\s]', '', new_text)  # remove punc
        new_text = re.sub('<unk>', '', new_text)  # remove numbers
        new_text = re.sub(r'\d+', '', new_text)  # remove numbers
        new_text = re.sub(' +', ' ', new_text)  # remove extra space
        new_text = re.sub('LRB|LSB|RSB|RRB', '', new_text)  # remove extra space
        new_text = new_text.lstrip()  # remove leading whitespaces

        if new_text != '':
            clean_data.append(new_text)

    return clean_data

def remove_continuous_duplicate(input_string):
    word_index = []
    input_words = input_string.split()
    for i in range(len(input_words) - 1):
        if (input_words[i] == input_words[i + 1]):
            word_index.append(i)

    for index in word_index:
        input_words.pop(index)

    return ' '.join(input_words)


def ne_string_hash(docs):
    result = []
    for doc in docs.to_dict()[0]:
        if (len(doc['multi_ner'][0]) > 1):
            if (get_last_word(doc['multi_ner'][0]) == 'PERSON'):
                result.append(doc['text'])

    return result
class NER:
    def get_NER_dict(self):
        df_train = pd.read_csv('../data/data.csv'.format(str(set)))
        df_test_1 = pd.read_csv('../data/FEVER_1_Test.csv'.format(str(set)))
        df_test_2 = pd.read_csv('../data/FEVER_2_Test.csv'.format(str(set)))
        df = pd.concat([df_train, df_test_1, df_test_2], axis=0)
        # df = df.iloc[0:100,:]
        df_claims = df['claims'].values.tolist()
        df_class = df['classes'].values.tolist()
        df_claims = remove_special_char(df_claims)

        file_positive_ner = {}
        file_negative_ner = {}
        common_ner = {}
        print('Collecting NERs')

        for i in range(len(df_claims)):
            text_data = df_claims[i]
            class_lable = df_class[i]
            doc = nlp(text_data)

            if(class_lable == 0):
                for ent in doc.ents:
                    if(ent.type not in file_negative_ner):
                        file_negative_ner[ent.type] = [ent.text]
                    else:
                        if (ent.text not in file_negative_ner[ent.type]):
                            file_negative_ner[ent.type].append(ent.text)

            else:
                for ent in doc.ents:
                    if (ent.type not in file_positive_ner):
                        file_positive_ner[ent.type] = [ent.text]
                    else:
                        if(ent.text not in file_positive_ner[ent.type]):
                            file_positive_ner[ent.type].append(ent.text)

            if(i%1000 == 0):
                print('Done: ',i)

        ner_normalized = {}
        ner_count = 1000000
        for key in file_positive_ner.keys():
            if(key not in file_positive_ner):
                continue
            else:
                positive_values = file_positive_ner[key]

            if(key not in file_negative_ner):
                continue
            else:
                negative_values = file_negative_ner[key]

            for value in positive_values:
                if(value in negative_values):
                    if(key not in common_ner):
                        common_ner[key] = [value]
                    else:
                        common_ner[key].append(value)
                    positive_values.remove(value)
                    negative_values.remove(value)

            file_positive_ner[key] = positive_values
            file_negative_ner[key] = negative_values


            if(len(common_ner.keys())>0):
                for key in common_ner.keys():
                    common_values = common_ner[key]
                    for values in common_values:
                        temp = key[0]+'-'+str(ner_count)
                        ner_count += 1
                        ner_normalized[values] = temp


                    unique_positive = file_positive_ner[key]
                    unique_negative = file_negative_ner[key]

                    if(len(unique_negative) > len(unique_positive)):
                        for i in range(len(unique_positive)):
                            ner_normalized[unique_positive[i]] = key[0]+'-'+str(ner_count)
                            ner_normalized[unique_negative[i]] = key[0]+'-'+str(ner_count)
                            ner_count += 1
                            if (i % 1000 == 0):
                                print('Done 1: ', i)


                        while(i < len(unique_negative)-1):
                            ner_normalized[unique_negative[i]] = key[0]+'-'+str(ner_count)
                            ner_count += 1
                            i += 1
                    else:
                        for i in range(len(unique_negative)):
                            ner_normalized[unique_positive[i]] = key[0]+'-'+str(ner_count)
                            ner_normalized[unique_negative[i]] = key[0]+'-'+str(ner_count)
                            ner_count += 1

                            if (i % 100 == 0):
                                print('Done 2: ', i)

                        while(i < len(unique_positive)-1):
                            ner_normalized[unique_positive[i]] = key[0]+'-'+str(ner_count)
                            ner_count += 1
                            i += 1
            else:
                unique_positive = file_positive_ner[key]
                unique_negative = file_negative_ner[key]
                i = 0
                while (i < len(unique_positive) - 1):
                    ner_normalized[unique_positive[i]] = key[0] + '-' + str(ner_count)
                    ner_count += 1
                    i += 1

                i = 0
                while (i < len(unique_negative) - 1):
                    ner_normalized[unique_negative[i]] = key[0] + '-' + str(ner_count)
                    ner_count += 1
                    i += 1
        with open("../../data/ner_normalized.json", "w") as outfile:
            json.dump(ner_normalized, outfile)

        #return ner_normalized