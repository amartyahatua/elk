import re
import nltk
import spacy
import stanza
import pandas as pd
from tqdm import tqdm



#stanza.download('en')
nlp = stanza.Pipeline('en')

# NE:
# Method 1: No change
# Method 2: Replace with explanation
# Method 3: Replace with POS
# Method 4: Attach explanation
# Method 5: Attach POS

# REST of the words:
# Method 1: No change
# Method 2: Replace with POS
# Method 3: Attach POS

def get_last_word(word):
    return word.split('-')[-1]

def remove_special_char(df):
    clean_data = []

    for x in df:
        new_text = re.sub('<.*?>', '', x)  # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text)  # remove punc
        new_text = re.sub('<unk>', '', new_text)
        new_text = re.sub(r'\d+', '', new_text)  # remove numbers
        new_text = re.sub(' +', ' ', new_text)  # remove extra space
        new_text = re.sub('LRB|LSB|RSB|RRB', '', new_text)  # remove extra space
        new_text = new_text.lstrip()  # remove leading whitespaces
        new_text = new_text.replace("\n", " ")
        new_text = new_text.lower()

        if new_text != '':
            clean_data.append(new_text)

        #return new_text

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

class Process_NE:
    # Method 1: No change
    def ne_no_change(self, docs):
        result_dict = {}
        for doc in docs.to_dict()[0]:
            if (len(doc['multi_ner'][0]) > 1):
                result_dict[doc['text']] = doc['text']
        return  result_dict

    # 2: Replace with explanation
    def ne_replace_explanation(self, docs):
        result_dict = {}
        for doc in docs.to_dict()[0]:
            if (len(doc['multi_ner'][0]) > 1):
                result_dict[doc['text']] = get_last_word(doc['multi_ner'][0])
        return result_dict

    # 3: Replace with POS
    def ne_replace_pos(self, docs):
        result_dict = {}
        for doc in docs.to_dict()[0]:
            if (len(doc['multi_ner'][0]) > 1):
                result_dict[doc['text']] = doc['xpos']
        return result_dict


    # 4: Attach explanation
    def ne_attach_explanation(self, docs):
        result_dict = {}
        for doc in docs.to_dict()[0]:
            if (len(doc['multi_ner'][0]) > 1):
                result_dict[doc['text']] = doc['text']+'#'+get_last_word(doc['multi_ner'][0])
        return result_dict

    # 3: Attach with POS
    def ne_attach_pos(self, docs):
        result_dict = {}
        for doc in docs.to_dict()[0]:
            if (len(doc['multi_ner'][0]) > 1):
                result_dict[doc['text']] = doc['text']+'#'+doc['xpos']
        return result_dict

    def ne_string_hash(self, docs, bst_person):
        result_dict = {}
        for doc in docs.to_dict()[0]:
            if (len(doc['multi_ner'][0]) > 1):
                if(get_last_word(doc['multi_ner'][0]) == 'PERSON'):
                    if (bst_person.search_person(bst_person.head, doc['text']) == False):
                        bst_person.insert(get_last_word(doc['text']))
                        bst_person.search_person(bst_person.head, doc['text'])
                        path_string = ''.join(bst_person.path)
                        result_dict[doc['text']] = path_string
                    else:
                        bst_person.search_person(bst_person.head, doc['text'])
                        path_string = ''.join(bst_person.path)
                        result_dict[doc['text']] = path_string
                    bst_person.path = []
                else:
                    result_dict[doc['text']] = doc['text']


        return result_dict


class Process_REST:
    # REST of the words:
    # Method 1: No change
    def rest_no_change(self, sentence):
        result_dict = {}
        input_doc = nlp(sentence)
        for doc in input_doc.to_dict()[0]:
            if (len(doc['multi_ner'][0]) == 1):
                result_dict[doc['text']] = doc['text']
        return  result_dict

    # 2: Replace with POS
    def rest_replace_pos(self, sentence):
        result_dict = {}
        input_doc = nlp(sentence)
        for doc in input_doc.to_dict()[0]:
            if (len(doc['multi_ner'][0]) == 1):
                result_dict[doc['text']] = doc['xpos']
        return result_dict

    # 3: Attach POS
    def rest_attach_pos(self, sentence):
        result_dict = {}
        input_doc = nlp(sentence)
        for doc in input_doc.to_dict()[0]:
            if (len(doc['multi_ner'][0]) == 1):
                result_dict[doc['text']] = doc['text'] +'#'+doc['xpos']
        return result_dict

nes = Process_NE()
rest = Process_REST()


def preprocess_method_1_1(df_claims):
    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if(count %10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_no_change(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if(len(rest_words_sent)> 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result


# NE 1, REST 2:
# Output: Ukrainian Soviet NN JJ VBG NN UN
def preprocess_method_1_2(df_claims):
    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_no_change(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 1, REST 3:
def preprocess_method_1_3(df_claims):
    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_no_change(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 2, REST 1:
def preprocess_method_2_1(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        doc = nlp(text_data)
        ner_dict = nes.ne_replace_explanation(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 2, REST 2:
def preprocess_method_2_2(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_replace_explanation(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 2, REST 3:
def preprocess_method_2_3(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_replace_explanation(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 3, REST 1:
def preprocess_method_3_1(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_replace_pos(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 3, REST 2:
def preprocess_method_3_2(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_replace_pos(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 3, REST 3:
def preprocess_method_3_3(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_replace_pos(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 4, REST 1:
def preprocess_method_4_1(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_attach_explanation(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 4, REST 2:
def preprocess_method_4_2(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_attach_explanation(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 4, REST 3:
def preprocess_method_4_3(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_attach_explanation(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 5, REST 1:
def preprocess_method_5_1(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_attach_pos(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 5, REST 2:
def preprocess_method_5_2(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_attach_pos(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# NE 5, REST 3:
def preprocess_method_5_3(df_claims):

    #df_claims = df_claims['claims'].values.tolist()
    df_claims = remove_special_char(df_claims)
    result = []
    count = 0
    for text_data in tqdm(df_claims):
        count += 1
        if (count % 10000 == 0):
            print('Preprocess done = ', count)
        rest_words = []
        text_data = text_data.strip()
        doc = nlp(text_data)
        ner_dict = nes.ne_attach_pos(doc)

        for word in text_data.split():
            if ((ner_dict != None) and (word not in ner_dict)):
                rest_words.append(word)

        rest_words_sent = ' '.join(rest_words)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        text_no_namedentities = []

        for token in doc.to_dict()[0]:
            try:
                if ner_dict != None and token['text'] in ner_dict:
                    text_no_namedentities.append(ner_dict[token['text']])
                else:
                    text_no_namedentities.append(res_dict[token['text']])
            except:
                continue
        result.append(" ".join(text_no_namedentities))
    return result

# def preprocess_method_6_1(df_claims, bst_person):
#     df_claims = df_claims['claims'].values.tolist()
#     df_claims = remove_special_char(df_claims)
#     result = []
#     for text_data in df_claims:
#         rest_words = []
#         text_data = text_data.strip()
#         doc = nlp(text_data)
#         ner_dict = nes.ne_string_hash(doc, bst_person)
#
#         for word in text_data.split():
#             if ((ner_dict != None) and (word not in ner_dict)):
#                 rest_words.append(word)
#
#         rest_words_sent = ' '.join(rest_words)
#         if (len(rest_words_sent) > 1):
#             res_dict = rest.rest_no_change(rest_words_sent)
#
#         text_no_namedentities = []
#
#         for token in doc.to_dict()[0]:
#             try:
#                 if ner_dict != None and token['text'] in ner_dict:
#                     text_no_namedentities.append(ner_dict[token['text']])
#                 else:
#                     text_no_namedentities.append(res_dict[token['text']])
#             except:
#                 continue
#
#         result.append(" ".join(text_no_namedentities))
#     print('Size = ', bst_person.getSize())
#     print(bst_person)
#     return result
#
#
# input_sentence = ["Barack Obama was born in beautiful Hawaii"]
# # doc = nlp(input_sentence)
# # print(doc)
#
# # data = pd.read_csv('../ml_model/preprocess_data/set_1/data.csv')
# # input_sentence = data
# print(preprocess_method_1_1(input_sentence))
# print(preprocess_method_1_2(input_sentence))
# print(preprocess_method_1_3(input_sentence))
# print("\n\n")
# print(preprocess_method_2_1(input_sentence))
# print(preprocess_method_2_2(input_sentence))
# print(preprocess_method_2_3(input_sentence))
# print("\n\n")
# print(preprocess_method_3_1(input_sentence))
# print(preprocess_method_3_2(input_sentence))
# print(preprocess_method_3_3(input_sentence))
# print("\n\n")
# print(preprocess_method_4_1(input_sentence))
# print(preprocess_method_4_2(input_sentence))
# print(preprocess_method_4_3(input_sentence))
# print("\n\n")
# print(preprocess_method_5_1(input_sentence))
# print(preprocess_method_5_2(input_sentence))
# print(preprocess_method_5_3(input_sentence))
# print("\n\n")
# print(preprocess_method_6_1(input_sentence))
# print(preprocess_method_6_2(input_sentence))
# print(preprocess_method_6_3(input_sentence))