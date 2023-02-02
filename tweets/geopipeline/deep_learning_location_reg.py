# -*- coding: utf-8 -*- 

from transformers import BertForTokenClassification, BertTokenizer
import torch
import lmr
import spacy
from time import gmtime, strftime
import sys
import json
import os
from more_itertools import chunked
from tqdm import tqdm
import re
from cleantext import clean


def deEmojify2(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r' ',text)


def preprocess_tweet_text(text):
    #print("Orginal: " + text)
    
    text = text.strip()
    # remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    # remove usernames
    text = re.sub('@[^\s]+', ' ', text)
    # remove the # in #hashtag
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # remove emojies
    text = deEmojify(text)
    text = deEmojify2(text)
    text = text.replace('_',' ')
    text = text.replace('\n',' ')
    text = text.replace('\t',' ')
    text = text.replace('\r',' ')
    text = text.replace('"',' ')
    text = text.replace('~',' ')
    text = text.replace('|',' ')
    text = text.replace(',','<<<<comma>>>>')


    preprocessed_text = clean(text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=False,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=True,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=True,      # replace all currency symbols with a special token
        no_punct=True,                 # fully remove punctuation
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_currency_symbol="",
        # replace_with_punct=" ",
        # lang="en"                       # set to 'de' for German special handling
    )

    # text = text.translate(constants.PUNCT_TRANSLATE_UNICODE)
    text = text.replace('<<<<comma>>>>',',')
    
    #remove extra spaces
    text = re.sub(' +', ' ', text)


    #print("Final: "+ text)

    # to avoid cases like ', '
    #to avoid cases like ' ', 'a', ',' etc
    #to avoid cases like ' E',
    if(len(text.replace(',','')) > 1 and len(text.replace(',','').replace(' ','')) > 1 and len(text.replace(' ','')) > 1 and len(text) > 1 and text != '' and text != ' '):
        return text
    else:
        return ''

def clean_custom(string):
    return string.replace("'", "").replace("\t", " ").replace("\n", " ").replace("\r", '').replace('"', '').strip()



# This function is for inserting a value in a dictionary
# INPUTS: dict, key
# OUTPUTS: the changed dictionary
def insert_in_dict(dictionary, key, value):
    # check key
    # print key
    # print value
    if key in dictionary:
      dictionary[key] = value
    else:
      if key != None:
          dictionary[key] = value

    return dictionary


model_path = "rsuwaileh/IDRISI-LMR-EN-timebased-typeless"
model = BertForTokenClassification.from_pretrained(model_path, output_loading_info=False)
tokenizer = BertTokenizer.from_pretrained(model_path)
#model = quantize_dynamic(org_model, {nn.Linear}, dtype=torch.qint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = model.to(device)

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
BATCH_SIZE = 5000

lines = []
lmr_mode = "TL"
ids = []


opened_file = open(sys.argv[1],encoding="UTF-8")

output_file = open(sys.argv[1]+'_candidates_final.json', 'w+',encoding="UTF-8")

for line in tqdm(opened_file):
    data = json.loads(line)

    if(data['_source'].get('tweet_details') == None or data['_source'].get('tweet_details').get('full_text') == None):
        print(data['_id'])
        pass
    else:
        string_for_bert = preprocess_tweet_text(clean_custom(data['_source']['tweet_details']['full_text']).encode("ascii", "ignore").decode('utf-8'))
        if(len(string_for_bert) < 3):
            pass
        else:
            lines.append(string_for_bert)
            ids.append(data['_id'])


# length_items = len(lines)
# batch_iter = int(length_items/BATCH_SIZE)

# print(batch_iter)

lines = list(chunked(lines, BATCH_SIZE))
ids = list(chunked(ids, BATCH_SIZE))

# print(lines)
# print(ids)
try:
    for i in tqdm(range(0,len(lines))):
    # for i in range(0,len(lines)):

        lms = []
        tokens = []

        # try:
            # print(len(lines[i]))
        tokens, lms = lmr.get_locations(lines[i], model, device, lmr_mode)
        # print(lms)

        for k in range(0, len(lms)):
            #print(lms[i])
            items = []
            json_item = {}
            data_json = {}
            # print("one")
            # print(lms[k])


            for t in range(0, len(lms[k])):
                tup1 = lms[k][t].replace(',', '').replace('(', ' ').replace(')', ' ').replace('.', '')
                tup2 = 'GPE'
                items.append((tup1,tup2,))

            insert_in_dict(json_item, 'ner_text_translated', items)
            insert_in_dict(json_item, 'tweet_id', ids[i][k])
            insert_in_dict(json_item, 'full_text', lines[i][k])
            data_json = json.dumps(json_item, indent=None, separators=(',',':'))

            output_file.write(data_json+'\n')



        # except Exception as ex:
        #     print(ex)
        #     pass

        output_file.flush()
except Exception as ex:
    print(ex)

        




    # 






