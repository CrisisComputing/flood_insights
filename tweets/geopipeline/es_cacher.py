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
# import constants

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


# def nominatim_or_es(call_type, query_term, tweet_id, lat, lon, es_location_object,misc_log):


def insert_search_call(call_type, origin, query_term, tweet_id, response, elastic_connection, misc_log):
    
    insert_dict = {}

    if(call_type == 'geo_call'):
        index_to_insert = "location_index_geo"
        response = json.dumps(response, indent=None, separators=(',',':'))

    if(call_type == 'gender_call'):
        index_to_insert = "gender_index"

    if(call_type == 'text_call'):
        index_to_insert = "location_index_complete"
        insert_dict['attribute'] = origin


    insert_dict['query_term'] = query_term
    insert_dict['response'] = response
    #origin tweet id
    insert_dict['origin_tweet_id'] = tweet_id

    insert_in_dict(insert_dict, "counter", 1)


    # data_json = json.dumps(insert_dict, indent=None, separators=(',',':'))
    # print(insert_dict)
    try:
         # es_object.index(index=index_name, body=data_json, id= data['id_str'])
        response = elastic_connection.index(index=index_to_insert, body=insert_dict, id= query_term)
        


    except Exception as ex:
        misc_log.warning("es location_index inserting issue..")
        misc_log.warning(ex)
        return

    insert_dict.clear()
    return

def get_search_call(call_type, query_term, elastic_connection, misc_log):

    if(query_term == ''):
        return []


    try:
        if(call_type == 'geo_call'):
            index_to_query = "location_index_geo"
        if(call_type == 'gender_call'):
            index_to_query = "gender_index"
        if(call_type == 'text_call'):
            index_to_query = "location_index_complete"


        s = Search(using=elastic_connection, index=index_to_query).filter("term", query_term=query_term)
        #s = s[0:1]
        response = s.execute()

        check_update = elastic_connection.update(index=index_to_query,id=query_term , body={"script": 'ctx._source.counter+=1'})
        #print(response)
        #print(response)
    except Exception as ex:
        # misc_log.warning("es location_index fetching issue..")
        # misc_log.warning(ex)
        return "not_in_location_index"

    #print(response.hits)

    if(len(response.hits) == 0):
        # print(response.hits)
        # print("hrererer")
        return "not_in_location_index"
    else:
        # print("here")
        #print(response)
        # constants.ES_HITS = constants.ES_HITS + 1

        if(call_type == 'geo_call'):

            return eval(response.hits[0].response)

        elif(call_type == 'gender_call'):

            return response.hits[0].response

        else:

            for i in range(0, len(response.hits)):
                #print(query_term)
                if(query_term == response.hits[i].query_term):
                    if(response.hits[i].response != 'nominatim_server_error'):
                        item_evaluated = eval(response.hits[i].response)
                        return item_evaluated
                    else:
                        []

            

            return 'not_in_location_index'


