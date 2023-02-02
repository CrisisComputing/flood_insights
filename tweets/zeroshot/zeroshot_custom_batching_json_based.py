from transformers import pipeline
import pandas as pd
import re, torch, os, json
import sys
import csv
from tqdm import tqdm
import emoji
import time, numpy as np
from cleantext import clean
from modules.common_utils import cacher, redis_connection, yml_reader, latency_cal

# import logging

# tf.get_logger().setLevel(logging.ERROR)

BATCH_SIZE_CLASSIFIER = 1

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device = 0, max_seq_len = 280, batch_size = 30)

# print(str(classifier))
# sys.exit()
# taxono = {'food_request': ['need food', 'need for food', 'food request', 'requesting food', 'urgent demand of food', 'urgent need of food', 'food needs', 'want food', 'food demands', 'demand for food', 'urgent demand for food', 'require food', 'appeal for food'], 'bread_request': ['need bread', 'need for bread', 'bread request', 'requesting bread', 'urgent demand of bread', 'urgent need of bread', 'bread needs', 'want bread', 'demand for bread', 'urgent demand for bread', 'require bread', 'appeal for bread'], 'milk_request': ['need milk', 'need of milk', 'milk request', 'requesting milk', 'urgent demand of milk', 'urgent need of milk', 'milk needs', 'want milk', 'demand for milk', 'urgent demand for milk', 'require milk', 'need formula milk', 'appeal for milk'], 'drinking_water_request': ['need drinking water', 'need of drinking water', 'drinking water request', 'requesting drinking water', 'urgent demand of drinking water', 'urgent need of drinking water', 'request for drinking water', 'want drinking water', 'require drinking water', 'appeal for drinking water'], 'blanket_request': ['need blanket', 'need of blanket', 'urgent demand of blanket', 'urgent need of blanket', 'request for blanket', 'appeal for blanket'], 'clothing_request': ['need clothing', 'need of clothing', 'need warm socks', 'urgent need of shoes', 'urgent demand of clothing', 'urgent need of clothing', 'request for clothing', 'require clothing', 'appeal for clothing'], 'mattress_request': ['need mattress', 'need of mattress', 'mattress request', 'urgent demand of mattress', 'urgent need of mattress', 'request for mattress', 'require mattress', 'appeal for mattress'], 'hygiene_items_request': ['need hygiene items', 'need of hygiene items', 'hygiene items request', 'request hygiene items', 'urgent demand of hygiene items', 'urgent need of hygiene items', 'request for hygiene items', 'require hygiene items', 'need infant diaper', 'appeal for hygiene items'], 'mosquito_net_request': ['need mosquito net', 'need of mosquito net', 'mosquito net request', 'requesting mosquito net', 'urgent demand of mosquito net', 'urgent need of mosquito net', 'request for mosquito net', 'require mosquito net', 'appeal for mosquito net'], 'utensils_request': ['need utensils', 'need of utensils', 'requesting utensils', 'urgent demand of utensils', 'urgent need of utensils', 'request for utensils', 'need kitchen sets', 'need stove', 'require utensils'], 'money_request': ['need money', 'requesting cash', 'requesting money', 'in need of money', 'demand of money', 'asking for money', 'appeal for money', 'need of cash'], 'shelter_request': ['requesting shelter', 'need shelter', 'asking for shelter', 'need for shelter', 'accommodation required', 'need accommodation', 'need tent', 'appeal for shelter', 'seeking shelter', 'tent required'], 'medical_assistance_request': ['need urgent medical assistance', 'medical assistance request', 'appeal for medical assistance', 'ambulance required', 'request for ambulance', 'request for medicine', 'need medicine', 'need healthcare worker', 'need healthcare facilities', 'asking for medical assistance', 'need urgent nurse', 'require nurse', 'shortage of nurse', 'need of nurse', 'need urgent doctor', 'require doctor', 'shortage of doctors', 'appeal for doctors'], 'medical_equipment_request': ['need medical equipment', 'medical equipment required', 'requesting medical equipment', 'medical equipment request', 'need of medical equipment', 'request for wheelchair', 'requesting wheelchair', 'request for crutches', 'appeal for medical equipment', 'appeal for wheelchair', 'appeal for crutches'], 'request_first_aid_kit': ['need first aid kit', 'first aid kit required', 'requesting first aid kit', 'first aid kits request', 'shortage of first aid kits', 'need of first aid kits', 'lack of first aid kits', 'appeal for first aid kits'], 'request_gloves': ['need gloves', 'gloves required', 'requesting gloves', 'gloves request', 'need of gloves', 'lack of gloves', 'shortage of gloves', 'appeal for gloves'], 'request_mask': ['need mask', 'mask required', 'requesting mask', 'mask request', 'need of mask', 'lack of masks', 'shortage of mask', 'appeal for masks'], 'rescue_request': ['asking for rescue', 'rescue request', 'urgent rescue request'], 'donation_request': ['asking for donation', 'donation request', 'donations required', 'urgent donations required', 'need for donations'], 'volunteer_need': ['volunteer request', 'need volunteer', 'need of volunteer', 'volunteer required', 'manpower required', 'request for volunteering efforts'], 'electricity_power_request': ['asking for electricity', 'electricity power request', 'electricity required', 'urgent electricity request', 'need for electricity'], 'donation_offer': ['offering donation', 'donation offers', 'give donations', 'present donations', 'donating drinking water', 'donating bread', 'donating milk', 'donating money', 'donating baby formula milk', 'donations provided'], 'food_offer': ['offer food', 'supplied food', 'supplying food', 'donating food'], 'drinking_water_offer': ['offer drinking water', 'supplied drinking water', 'supply drinking water'], 'bread_offer': ['offer bread', 'supplied bread', 'supply bread'], 'milk_offer': ['offer milk', 'supplied milk', 'supply milk'], 'money_offer': ['offer money', 'provide cash', 'offer cash', 'supply money'], 'blanket_offer': ['offer blanket', 'provide blanket', 'supply for blanket'], 'clothing_offer': ['offer clothing', 'offer warm clothing', 'offer warm socks', 'offer shoes', 'provide clothing', 'supply clothing'], 'mattress_offer': ['offer mattress', 'supply mattress', 'provide mattress'], 'hygiene_items_offer': ['supply hygiene items', 'supply hygiene items', 'provide hygiene items', 'provide infant diaper', 'offer infant diaper'], 'mosquito_net_offer': ['offer mosquito net', 'provide mosquito net', 'supply mosquito net'], 'utensils_offer': ['offer utensils', 'supply utensils', 'provide utensils', 'provide kitchen sets', 'supplu kitchen sets', 'supply stove', 'provide stove', 'supply stove'], 'medical_assistance_offer': ['offer medical assistance', 'medical assistance offer', 'free healthcare offer', 'provided medical assistance', 'doctors available', 'doctor volunteering', 'available doctor', 'nurse available', 'available nurse', 'nurses volunteering'], 'medical_equipment_offer': ['offer medical equipment', 'medical equipment offer', 'medical equipment available', 'crutches available', 'wheelchair available'], 'mask_offer': ['offer mask', 'mask offer', 'mask available'], 'glove_offer': ['offer gloves', 'gloves offer', 'gloves available'], 'shelter_offer': ['offer shelter', 'provide shelter', 'offer accommodation', 'provide accommodation', 'offer tent', 'provide tent'], 'volunteer_offer': ['available to volunteer', 'ready to volunteer', 'offer to help', 'willing to help', 'willing to volunteer'], 'infrastructure_damage': ['building damage', 'bridge damage', 'bridge destroyed', 'building destroyed', 'destroyed road', 'road damage', 'destroyed building', 'damage dam', 'damage school', 'damage hospital', 'destroyed school', 'destroyed hospital', 'railway track damage', 'railway track destroyed', 'house damage', 'house destroyed'], 'water_system_and_sewage_damages': ['drainage problem', 'water system damage', 'sewage damage', 'damage to sewage system', 'water pipe damage', 'sewage blockage', 'blocked sewage', 'water drainage system damage', 'dam damage', 'water reservoir damage'], 'utility_damage': ['power outage', 'electricity outage', 'power failure', 'internet inaccessible', 'electricity pole damage', 'destroyed communication', 'internet problem', 'internet outage', 'mobile network damage', 'mobile network destroyed', 'telecommunication system damage', 'telecommunication system destroyed', 'electricity failure'], 'agriculture_and_forestry': ['land affected', 'crops damaged', 'irrigation destroyed', 'damaged irrigation system', 'crops destroyed', 'irrigation system damage', 'landscape damage', 'forest affected', 'destroyed trees', 'jungle affected'], 'livestocks_fisheries': ['fishes affected', 'fishery destroyed', 'dead fish', 'fish losses', 'dead animals', 'dead herd', 'affected herd', 'cattle affected', 'affected animal'], 'affected_person': ['dead person', 'injured person', 'injured people', 'dead people', 'trapped person', 'people killed', 'became homeless', 'people lost life', 'deceased person', 'casualties', 'deceased people'], 'missing_person': ['missing person', 'missing individual', 'lost and found people', 'missing child', 'missing relatives', 'lost children', 'lost parent', 'search for missing person', 'looking for missing people', 'lost family'], 'weather_information_and_updates': ['weather report', 'weather information and update', 'weather forecast', 'rain forecast', 'rain prediction', 'storm prediction', 'storm forecast', 'rainfall information'], 'health_and_disease_update': ['disease and illness', 'mental health', 'emotional wellbeing', 'physical wellbeing', 'sickness report', 'disease outbreak', 'health issues', 'disease report', 'depression'], 'sanitation_and_hygiene_updates': ['access to sanitation facilities', 'availability of clean water', 'clean water access', 'waste disposal', 'sewage disposal', 'water contamination', 'lack safe drinking water', 'access to toilets', 'access to latrines', 'access to handwashing facilities', 'pollution'], 'caution_advice_and_other_public_information': ['safety advice', 'warning issued', 'warning lifted', 'evacuation instructions', 'evacuation alerts', 'emergency measure', 'emergency broadcasts', 'emergency alerts', 'please be advised'], 'sympathy_and_prayers': ['sympathy', 'emotional support', 'prayers', 'thoughts and prayers', 'empathy', 'kind prayer messages', 'heartfelt prayers'], 'logistics_and_transportation': ['goods and supplies delivery', 'goods storage', 'delivery of goods', 'storage of supplies', 'goods delivery', 'logistics', 'transportation delay'], 'complaints': ['food complaints', 'complaint', 'dissatisfaction', 'donation complaints', 'shelter complaints', 'funds complaints', 'corruption complaints', 'mismanagement', 'management complaints', 'resources complaints', 'sewage complaints', 'protest', 'insurance complaints', 'criticize'], 'security_and_concerns': ['safety concern', 'security concern', 'feeling insecure', 'robberies or theft reports', 'under attack', 'violence reports', 'feeling unsafe'], 'insurance': ['insurance availability', 'insurance', 'insurance claim', 'insurance unavailability', 'insurance issues']}
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


regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def remove_extra_lines(text):
    clean_text = os.linesep.join([s for s in text.splitlines() if s])
    return clean_text


def func_thresh(seqs, labs, scores, threshold, taxono):
    '''
    :param seqs: input tweet text, (not used in processing now but i was using it earlier)
    :param labs: The zershot_label column of tsv file accepted as a list
    :param scores: The zeroshot_label_confidence column of tsv accepted as a list
    :param threshold: Threshold over the confidence score
    :return: Mapped classes with weighted score
    '''
    data = []
    data_2 = []
    for (seq, lab, score) in zip(seqs, labs, scores):
        # print(seq)
        in_data = {}
        in_data_high = {}
        for keys, _ in taxono.items():
            for subkeys, _ in taxono[keys].items():
                for tax, val in taxono[keys][subkeys].items():
                        label_dumy = []
                        score_dumy = []
                        count = 0
                        for (l, s) in zip(lab, score):
                            if l:
                                s = float(s)
                                if s < threshold:
                                    break
                                l = l.lstrip('"').lstrip()
                                l = str(l.replace("'", ""))
                                if l in val:
                                    label_dumy.append(l)
                                    score_dumy.append(s)
                                    count += 1
                                    # break
                        if label_dumy:
                            in_data[tax] = np.mean(score_dumy)
                        # if label_dumy:
                        #     print(tax, label_dumy[:5], score_dumy[:5])
            if len(in_data) > 0:
                for midlevels in in_data:

                    midhigh = list(taxono[keys].keys())
                    for midh in midhigh:
                        if midlevels in taxono[keys][midh].keys():
                            scores_hl = np.array(list(in_data.values())).mean()
                            in_data_high[keys] = scores_hl

        in_data_2 = {k: v for k, v in sorted(in_data.items(), key=lambda item: item[1], reverse=True)}
        in_data_hl_2 = {k: v for k, v in sorted(in_data_high.items(), key=lambda item: item[1], reverse=True)}

        data.append(in_data_2)
        data_2.append(in_data_hl_2)
    return data, data_2

def check_word(x):

    preprocessed_text = clean(x,
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

    x_array = preprocessed_text.split(' ')

    if len(x_array) >= 2:
        return x
    else:
        return ''

def preproces_tweet(x):
    """
    :param x: input text
    :return: preproceed text
    """
    x = ' '.join([word for word in x.split(' ')])
    x = os.linesep.join([s for s in x.splitlines() if s]) # remove extra lines
    x = regrex_pattern.sub(r'', x) # remove emojis
    x = x.encode('ascii', 'ignore').decode() # remove any special chracters

    x = emoji.get_emoji_regexp().sub(u' ', x)

    x = deEmojify(x)
    x = deEmojify2(x)
    
    # Remove mentions
    x = re.sub("@\S+", "", x)
    # # Remove URL
    x = re.sub("https*\S+", "", x)
    # Remove ticks and the next character
    x = re.sub("\'\w+", '', x)
    # Replace the over spaces
    x = re.sub('\s{2,}', " ", x)

    if(len(x.replace(',','')) > 1 and len(x.replace(',','').replace(' ','')) > 1 and len(x.replace(' ','')) > 1 and len(x) > 1 and x != '' and x != ' '):
        return check_word(x)
    else:
        return ''
## Data Reading

def load_disaster_data(json_file):
    tweet_text = []
    tweet_id = []
    with open(json_file, "r") as f:
        for line in tqdm(f):
            try:
                l_dict = json.loads(line)
                id = l_dict["_id"]
                text = preproces_tweet(l_dict["_source"]["tweet_details"]["full_text_translated"])
                # text = preproces_tweet(l_dict["_source"]["text_full"])
                if len(text)<1:
                    print("Tweet with nothing left after preprocessing !, Excluding ",l_dict["_source"]["tweet_details"]["full_text_translated"]
                          , "------ after preprocessing", text)
                    continue
                tweet_id.append(id)
                tweet_text.append(text)

            except Exception as e:
                print(e)
                pass
    return tweet_text,tweet_id


print(sys.argv)
json_file = sys.argv[1]
out_file = sys.argv[1]+'_zeroshot_output.tsv'

if len(sys.argv) > 3:
    taxo_file = sys.argv[3]
else:
    taxo_file = 'taxonomy_singulars_only_2ndJan23.json'

candidate_labels = []
with open(taxo_file) as f:
    taxonomy_dict = json.load(f)
    for keys in taxonomy_dict:
        for subkeys in taxonomy_dict[keys]:
            for subsubkeys in taxonomy_dict[keys][subkeys]:
                for labels in taxonomy_dict[keys][subkeys][subsubkeys]:
                    candidate_labels.append(labels)
                    # print(labels)

# df = pd.read_csv('/export/sc-crisis-02/experiments/belmont_system/tweets/zeroshot/labels_full_v4_23Nov.csv')
# candidate_labels = df['labels'].tolist()
candidate_labels = [label.replace('_',' ').lower() for label in candidate_labels]
hypothesis_template = "The message in this text is related to {} "
#

# df_test = pd.read_excel('test_tweets.xlsx')
tweet_text,tweet_id = load_disaster_data(json_file)
print("Info : Total Tweets = ", len(tweet_text), "Candidate Labels :" , len(candidate_labels))
tic = time.time()
print("INFO : Starting inference on Dataset")
with open(out_file,'w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    writer.writerow(["tweet_id", "zeroshot_class_90", "zeroshot_class_95", "zeroshot_class_98",
                     "zeroshot_highlevel_90","zeroshot_highlevel_95","zeroshot_highlevel_98","zershot_label",
         "zeroshot_label_confidence"])
    latency = latency_cal.CalculateLatency(int(BATCH_SIZE_CLASSIFIER))
    print(latency)
    latency.start_time()

    for minibatch in range(0,len(tweet_text),BATCH_SIZE_CLASSIFIER):
        tweet_id_batch = tweet_id[minibatch:minibatch+BATCH_SIZE_CLASSIFIER]
        tweet_text_batch = tweet_text[minibatch:minibatch+BATCH_SIZE_CLASSIFIER]
        out = classifier(tweet_text_batch, candidate_labels, hypothesis_template=hypothesis_template,  multi_label=True)
        latency.end_time()
        latency.get_latency_throughput()
        seqs = []
        labs = []
        scores = []
        for it in out:
            # print(it)
            for key, val in it.items():
                if key == "sequence":
                    seqs.append(it[key])
                elif key == "labels":
                    labs.append(it[key])
                elif key == "scores":
                    scores.append(it[key])
        data90, high_level90 = func_thresh(seqs, labs, scores, 0.90, taxonomy_dict)
        data95, high_level95 = func_thresh(seqs, labs, scores, 0.95, taxonomy_dict)
        data98, high_level98 = func_thresh(seqs, labs, scores, 0.98, taxonomy_dict)

        for id_t, zl90,zl95,zl98,hl90,hl95,hl98, zlkw,zlc in zip(tweet_id_batch,
                                                data90,data95,data98,high_level90,high_level95,high_level98,
                                                 labs,scores):
            writer.writerow([id_t,zl90,zl95,zl98,hl90,hl95,hl98,zlkw,zlc])

        BATCH_SIZE_CLASSIFIER = BATCH_SIZE_CLASSIFIER * 2

        latency.batch_size = BATCH_SIZE_CLASSIFIER

# toc = time.time()
# print((toc-tic), " seconds")
