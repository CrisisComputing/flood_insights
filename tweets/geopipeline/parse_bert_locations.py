#!/usr/bin/python3.7.0

"""
   Created by Umair Qazi - 
   Research Assistant at 
   Qatar Computing Research Institute
   Dec 27, 2018
"""

# -*- coding: utf-8 -*- 
import json
import csv
import sys
import datetime  
import os
import logging
from geolib import geohash
import threading
from multiprocessing import Process
import sys
from ast import literal_eval
import logging
import random
import reverse_geocoder
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import urllib
import time
import urllib.request
import json
import collections
import es_cacher
from cleantext import clean
from copy import deepcopy
import re
import pandas as pd
import spacy
import collections
import emoji
import spacy
from elasticsearch import helpers
from tqdm import tqdm


SEARCH_URL = "http://localhost:7070/search?format=json&addressdetails=1&accept-language=en&limit=1&q="






###This function is for inserting a value in a dictionary
###INPUTS: dict, key
###OUTPUTS: the changed dictionary
def insert_in_dict(dictionary, key, value):
    #check key
    # key
    # value
    if key in dictionary:
        dictionary[key] = value
    else:
        if key != None:
            dictionary[key] = value

    return dictionary


def get_search_result(query_term, misc_log):
    try:
        array_json_response = urllib.request.urlopen(SEARCH_URL + urllib.parse.quote(query_term, encoding='utf-8')).read()
        #print(array_json_response)
        return array_json_response.decode("utf-8")

    except Exception as ex:
        misc_log.warning("nominatim search server issue..")
        misc_log.warning(query_term)
        misc_log.warning(ex)
        return "nominatim_server_error"

        



#Ensure that lat lon are always string, and not reversed
def get_geo_result(lat, lon, misc_log):

    try:
        json_response = urllib.request.urlopen(REVERSE_URL + lat + "&lon=" + lon).read()
        return literal_eval(json_response.decode("utf-8"))
    except Exception as ex:
        misc_log.warning("nominatim reverse server issue..")
        misc_log.warning(ex)
        return "nominatim_server_error"








def nominatim_or_es(call_type, origin, query_term, tweet_id, lat, lon, es_location_object, misc_log, gender_arr):

    #print('nominatim or es')
    #print(call_type)
    es_response = es_cacher.get_search_call(call_type, query_term, es_location_object, misc_log)
    #print(origin)
    #print(query_term)


    if(call_type == "geo_call"):
        try:
            if(es_response == "not_in_location_index"):
                nominatim_response = get_geo_result(lat, lon, misc_log)
                # constants.NOMINATIM_HITS += 1
                # evaluated_nominatim_response = eval(nominatim_response)

                es_cacher.insert_search_call(call_type, origin, query_term, tweet_id, nominatim_response, es_location_object, misc_log)

                
                return nominatim_response

            else:
                return es_response
        except Exception as ex:
            misc_log.warning(ex)

    elif(call_type == "gender_call"):
        try:
            if(es_response == "not_in_location_index"):
                #nominatim_response = get_geo_result(lat, lon, misc_log)first_name,model, vectorizer
                predicted_gender = gender_predictor.predict_gender(gender_arr[0], gender_arr[1], gender_arr[2])

                # constants.NOMINATIM_HITS += 1
                # evaluated_nominatim_response = eval(nominatim_response)

                es_cacher.insert_search_call(call_type, origin, query_term, tweet_id, predicted_gender, es_location_object, misc_log)

                
                return predicted_gender

            else:
                return es_response
        except Exception as ex:
            misc_log.warning(ex) 

    else:
        try:
            if(es_response == "not_in_location_index"):



                nominatim_response = get_search_result(query_term, misc_log)

                if(nominatim_response == "nominatim_server_error"):
                    return []


                # constants.NOMINATIM_HITS += 1
                #evaluated_nominatim_response = eval(nominatim_response)
                # print(nominatim_response)
                # print('**********')

                es_cacher.insert_search_call(call_type, origin, query_term, tweet_id, nominatim_response, es_location_object, misc_log)
                
                return eval(nominatim_response)

            else:
                return es_response
        except Exception as ex:
            misc_log.warning(ex)






def insert_address_in_dict(dictionary, nominatim_item):

        if('address' in nominatim_item):
            address = nominatim_item['address']
            
            insert_in_dict(dictionary, 'address', {})


            if('neighbourhood' in nominatim_item['address']):
                insert_in_dict(dictionary['address'], 'neighbourhood', nominatim_item['address']['neighbourhood'])

            if('city' in nominatim_item['address']):
                insert_in_dict(dictionary['address'], 'city', nominatim_item['address']['city'])

            if('county' in nominatim_item['address']):
                insert_in_dict(dictionary['address'], 'county', nominatim_item['address']['county'])

            if('state' in nominatim_item['address']):
                insert_in_dict(dictionary['address'], 'state', nominatim_item['address']['state'])

            if('country' in nominatim_item['address']):
                insert_in_dict(dictionary['address'], 'country', nominatim_item['address']['country'])

            if('country_code' in nominatim_item['address']):
                insert_in_dict(dictionary['address'], 'country_code', nominatim_item['address']['country_code'])



        if 'lat' in nominatim_item:
            insert_in_dict(dictionary, 'lat', literal_eval(nominatim_item['lat']))
        if 'lon' in nominatim_item:
            insert_in_dict(dictionary, 'lon', literal_eval(nominatim_item['lon']))

            geo_hash = geohash.encode(nominatim_item['lat'], nominatim_item['lon'], 12)
            insert_in_dict(dictionary, 'geohash', geo_hash)


        # insert_in_dict(dictionary, 'response', str(nominatim_item))


        return dictionary


def create_keywords_array(array, lang):
    # print(array)
    new_array = []
    for i in range(0, len(array)):
        if(array[i]["label"] == 'FAC' or array[i]["label"] == 'GPE' or array[i]["label"] == 'LOC'):
            new_array.append(array[i]["name"])

    return new_array



def create_dict_json(array):
    new_array = []
    for i in range(0, len(array)):
        new_array.append({'name':array[i][0], 'label':array[i][1]})

    return new_array



def create_classifier_dict_json(array):
    new_array = []
    for i in range(0, len(array)):
        new_array.append({'name':array[i][0], 'score':array[i][1]})

    return new_array



def parse_candidates(es_object, es_location_object, str_dict, json_dict, resolved_dict, major_dict, item_dict_response, candidates_array, candidate_log, gender_model_arr):
    
    case = 'none'
    

    #the query term and location discrepency 
    try:
        nominatim_candidate_array = candidates_array
        nominatim_calls = []
        nominatim_calls_query = []
        tweet_location_info_arr = []
        # print(nominatim_candidate_array)



        for k in range(0, len(nominatim_candidate_array)):

            try:
                if(nominatim_candidate_array[k].strip() != ''):

                    nominatim_return_arr = nominatim_or_es("text_call", str_dict, nominatim_candidate_array[k].strip(), json_dict['tweet_details']['id_str'], 0, 0, es_location_object, candidate_log, gender_model_arr)
                    # print(nominatim_return_arr)


                    #print(nominatim_return_arr)
                    if(type(nominatim_return_arr) != type(['a'])):
                        print(nominatim_candidate_array[k])
                        #print("location_module")

                        print(type(nominatim_return_arr))
                        # print(nominatim_return_arr)



                    if(nominatim_return_arr != None and len(nominatim_return_arr) > 0):
                        nominatim_calls.append(nominatim_return_arr[0])
                        nominatim_calls_query.append(nominatim_candidate_array[k])



                    # print(nominatim_return_arr)
                    # if(nominatim_return_arr != None and len(nominatim_return_arr) > 0):
                    #     for b in range(0, len(bigrams_returned)):
                    #         if(nominatim_candidate_array[k] in bigrams_returned[b]):
                    #             check = True

                    #     if(check == False):

                    


            except Exception as ex:
                candidate_log.warning("nominatim_candidate_array response issue..")
                candidate_log.warning(ex)
                candidate_log.warning(nominatim_return_arr)
                # print(len(nominatim_candidate_array))
                # candidate_log.warning(nominatim_candidate_array[k])
                # candidate_log.warning(nominatim_return_arr)
                pass

            
            # print(str(k) + '******************')
            # print(bigrams_returned)
            # print(nominatim_calls_query)
        # print(nominatim_calls)

        # if(str_dict == "tweet_text"):
        #     json_dict['tweet_details']['text_processed'] = str(nominatim_calls_query)
        #     json_dict['tweet_details']['text_processed_length'] = len(nominatim_calls_query)


        # if(str_dict == "user_location"):
        #     json_dict['tweet_details']['user_location_processed'] = str(nominatim_calls_query)
        #     json_dict['tweet_details']['user_location_processed_length'] = len(nominatim_calls_query)

        tweet_majority = []
        #some tweets might not have country_code
        if(len(nominatim_calls) > 0):

            # print("here")

            for t in range(0, len(nominatim_calls)):

                if 'country_code' in nominatim_calls[t]['address']:
 
                    nominatim_item = nominatim_calls[t]

                    item_dict_response = insert_address_in_dict({}, nominatim_item)

                    item_dict_response['query_term'] = nominatim_calls_query[t]
                    item_dict_response['response'] = str(nominatim_item)


                    tweet_location_info_arr.append(item_dict_response)

                    tweet_majority.append(nominatim_calls[t]['address']['country_code'])


            # insert_in_dict(json_dict['location_info'], 'tweet', tweet_location_info_arr)

            # print(tweet_majority)
            counter = collections.Counter(tweet_majority)

            if(len(counter.most_common()) !=0):

                try:

                    (cc_value, count) = counter.most_common()[0]
                    # case = str_dict

                except Exception as ex:
                    candidate_log.warning("Counter Issue...")
                    candidate_log.warning(tweet_location_info_arr)
                    candidate_log.warning(ex)
                    candidate_log.warning(counter)
                    candidate_log.warning(counter.most_common())
                    candidate_log.warning("\n")
                    pass
                # print(cc_value)
                
                for final in range(0, len(nominatim_calls)):
                    try:
                        if(nominatim_calls[final]['address'].get('country_code')!= None and 
                            nominatim_calls[final]['address']['country_code'] == cc_value):
                            major_dict = nominatim_calls[final]
                            query_term_final = nominatim_calls_query[final]
                            
                    except Exception as ex:
                        candidate_log.warning("nominatim_calls final issue...")
                        candidate_log.warning(ex)
                        candidate_log.warning(nominatim_calls[final]['address'])
                        pass

                if(len(major_dict) != 0):
                    resolved_dict = insert_address_in_dict({}, major_dict)

                    #fix query term for items
                    resolved_dict['query_term'] = query_term_final
                    
                    resolved_dict['response'] = str(major_dict)

                    case = str_dict
                    
                    if(str_dict == 'user_profile_description_location'):
                        insert_in_dict(json_dict['location_info'], 'resolved_location', resolved_dict)
                        insert_in_dict(json_dict['location_info'], str_dict, tweet_location_info_arr[0])
                    else:
                        insert_in_dict(json_dict['location_info'], 'resolved_location', resolved_dict)
                        insert_in_dict(json_dict['location_info'], str_dict, tweet_location_info_arr)



        return case
    except Exception as ex:
            candidate_log.warning("problem in content location parsing..")
            candidate_log.warning(ex)
            candidate_log.warning(json_dict['tweet_details'])
            candidate_log.warning(str_dict)
            resolved_dict.clear()
            major_dict.clear()
            item_dict_response.clear()
            return 'none'



        # 'full_name': 'İstanbul, wa, usa'
        # 'country': 'united states'

        # 'full_name': 'San Francisco, CA'
        # 'country': 'united states'

        #full_name + country

        #'full_name': 'Christiansted, Virgin Islands, U.S.'
        #'country': 'Virgin Islands, U.S.'

        #"Ismir, Turkey"
        #"San Francisco, CA"




def connect_elasticsearch(host, port):
    es = None
    es = Elasticsearch([{'host': host, 'port': port, 'timeout':10000, "maxsize": 500000, 'max_retries':1000, 'retry_on_timeout':True}])
    if es.ping():
      print('Connected')
    else:
      print('Could NOT connect!')
    return es


es_41 = connect_elasticsearch("10.4.2.41", 9500)
es_107 = connect_elasticsearch("10.4.2.107", 9500)
total_items_processed = 0
index_name = '220828090425_pakistan_floods_2022_image'
bulk_array = []

opened_file = open(sys.argv[1], encoding='utf-8')

for line in tqdm(opened_file):
    data = json.loads(line)

    ner_text_translated = data["ner_text_translated"]
    image_id = data["tweet_id"]

    json_dict = insert_in_dict({}, 'tweet_details', {})
    insert_in_dict(json_dict, 'location_info', {})

    insert_in_dict(json_dict['tweet_details'], 'id_str', image_id)

    # print(json_dict)

    tweet_text_array = create_dict_json(ner_text_translated)
    
    candidates_array = create_keywords_array(tweet_text_array, "en")

    # print(candidates_array)

    if(len(candidates_array) >= 1):
        parse_candidates(es_41, es_107, "tweet_text", json_dict, {}, {}, {}, candidates_array, None, [])

    data_json = json.dumps(json_dict, indent=None, separators=(',',':'))
    # print(data_json)
    
    doc = {}

    insert_in_dict(doc, 'location_info', json_dict['location_info'])




    bulk_array.append(
      {
      "_index": index_name, 
      "_id": image_id,
      "_op_type": "update",
      "doc_as_upsert": True, 
      "doc": deepcopy(doc),
      "retry_on_conflict": 5}
      )


    if(len(bulk_array) > 1000):
        # print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        try:
            res = helpers.bulk(es_41, bulk_array)
            print(total_items_processed)
            print(res)
            bulk_array = []
        except Exception as ex:
            print(ex)
            print("Bulk insertion issue...")


try:
    res = helpers.bulk(es_41, bulk_array)
    print(total_items_processed)
    print(res)
    bulk_array = []
except Exception as ex:
    print(ex)
    print("Bulk insertion issue...")





