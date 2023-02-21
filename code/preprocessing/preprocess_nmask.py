import re
import stanza
nlp = stanza.Pipeline('en')

# NE:
# Method 1: Replace NE with N-mask
# Method 2: Attach NE with N-mask
# Method 3: Attach NE with N-mask with Explaination
# Method 4: Attach N-mask with Explaination

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

class Process_NE:
    # 1: Replace NE with N-mask
    def nmask(self, df_claims, ner_normalized):
        result_dict = {}
        doc = nlp(df_claims)

        for ent in doc.ents:
            if (ent.text in ner_normalized):
                result_dict[ent.text] = ner_normalized[ent.text]
        return result_dict

    # 2: Attach NE with N-mask
    def ne_nmask(self, df_claims, ner_normalized):
        result_dict = {}
        doc = nlp(df_claims)

        for ent in doc.ents:
            if (ent.text in ner_normalized):
                result_dict[ent.text] = ent.text+'#'+ner_normalized[ent.text]
        return result_dict


    # 3: Attach NE with N-mask with Explaination
    def ne_nmask_explanation(self, df_claims, ner_normalized):
        result_dict = {}
        doc = nlp(df_claims)

        for ent in doc.ents:
            if (ent.text in ner_normalized):
                result_dict[ent.text] = ent.text+'#'+ner_normalized[ent.text]+'#'+ent.type
        return result_dict

    # 4: Attach N-mask with Explaination
    def nmask_explanation(self, df_claims, ner_normalized):
        result_dict = {}
        doc = nlp(df_claims)

        for ent in doc.ents:
            if (ent.text in ner_normalized):
                result_dict[ent.text] = ent.text+'#'+ner_normalized[ent.text]+'#'+ent.type
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


def preprocess_method_1_1(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []
    for text_data in df_claims:
        ner_dict = nes.nmask(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if(len(rest_words_sent)> 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)


        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue
        result.append(text_data)
    return result

def preprocess_method_1_2(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.ne_nmask(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)
        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if(len(rest_words_sent)> 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_1_3(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.ne_nmask_explanation(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_1_4(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.nmask_explanation(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_no_change(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result


def preprocess_method_2_1(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.nmask(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_2_2(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.ne_nmask(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_2_3(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.ne_nmask_explanation(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_2_4(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.nmask_explanation(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_replace_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_3_1(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.nmask(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_3_2(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.ne_nmask(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_3_3(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.ne_nmask_explanation(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue

        result.append(text_data)
    return result

def preprocess_method_3_4(df_claims, ner_normalized):
    df_claims = remove_special_char(df_claims)
    result = []

    for text_data in df_claims:
        ner_dict = nes.nmask_explanation(text_data, ner_normalized)

        rest_words_sent = text_data
        for key_val in ner_dict.keys():
            rest_words_sent = re.sub(key_val, '', rest_words_sent)

        rest_words_sent = re.sub(r'..', '', rest_words_sent)
        if (len(rest_words_sent) > 1):
            res_dict = rest.rest_attach_pos(rest_words_sent)

        for ner_key in ner_dict.keys():
            text_data = re.sub(ner_key, ner_dict[ner_key], text_data)

        for rst_key in res_dict.keys():
            try:
                text_data = re.sub(rst_key, res_dict[rst_key], text_data)
            except:
                continue
        result.append(text_data)
    return result