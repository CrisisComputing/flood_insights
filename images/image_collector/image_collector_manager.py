# -*- coding: utf-8 -*- 
import threading
import time
import sys
from modules.common_utils import cacher, redis_connection, yml_reader, helpers
from modules.common_utils.helpers import Dotdict, insert_in_dict
import json
import image_collector_classes
import image_cache_mapping

# from inf_place import resolve_place
# from inf_tweet_text import resolve_tweet_text
# from inf_user_description import resolve_user_description
from elasticsearch import Elasticsearch
import logging

def connect_elasticsearch(host, port):
    es = None
    es = Elasticsearch([{'host': host, 'port': port, 'timeout':10000, "maxsize": 500000}])
    if es.ping():
        print('Elasticsearch Connected')
    else:
        print('Elasticsearch Could NOT connect!')
    return es

def image_downloader(collection_code, tweet_obj, url_cache, path_to_download_imgs, response_dict):

    #print(url_cache)

    # threads_array = []
    image_downloader = image_collector_classes.TweetMediaExtractor(collection_code, tweet_obj, url_cache, path_to_download_imgs)
    #print(image_downloader)
    if(len(image_downloader.images) > 0):
        # print(image_downloader.images)
        response_dict['image_paths'] = image_downloader.images
        response_dict['tweets'] =  image_downloader.tweets

        #print(len(response_dict['tweets']))

    
    #Pass tweet object to get the appropriate attributes
    
    # threads_array.append(threading.Thread(target=resolve_coordinates,args=(location_info.data.get('tweet'), nominatim, location_cache, location_info)))

    # threads_array.append(threading.Thread(target=resolve_user_location,args=(location_info.data.get('tweet'), nominatim, location_cache, location_info)))

    # threads_array.append(threading.Thread(target=resolve_tweet_text,args=(location_info.data.get('ner'), nominatim, location_cache, location_info)))

    

    # threads_array.append(threading.Thread(target=resolve_place,args=(tweet_obj, nominatim, location_info)))


    # threads_array.append(threading.Thread(target=resolve_user_description,args=(ner_obj, nominatim, location_info)))

    # for i in range(0, len(threads_array)):
    #     threads_array[i].start()

    # for k in range(0, len(threads_array)):
    #     threads_array[k].join()

    # threads_array = []



#Loading the yml file
relative_path = sys.argv[1]
file_name = relative_path.split('/')[1].split('.')[0]
config = yml_reader.ConfigReader(relative_path)

#connecting to redis and 
redis = redis_connection.RedisConnection(config.loaded_file['redis']['host'], 
                                        config.loaded_file['redis']['port'],
                                        config.loaded_file['redis']['input_q'])

batch_size = config.loaded_file['downloader_thread_size']


#Creating the cache and passing the config details
cache_details = config.loaded_file['cache']

url_cache = cacher.Cacher(cache_details['host'], 
                    cache_details['port'])


#base_path + collection_code + monthly_path
path_to_download_imgs = config.loaded_file['path_to_download_imgs']

# tweet_info_q = config.loaded_file['tweet_info_q']


output_qs = config.loaded_file['redis']['output_qs']


es_object_cache = connect_elasticsearch(cache_details['host'], cache_details['port'])
es_object_dedup = connect_elasticsearch('10.4.2.100', 9200)

check_dict = {'stop':False}
check_dict = Dotdict(check_dict)

t1 = threading.Thread(target=helpers.stop_module,args=(check_dict,file_name))
t1.start()

#reset the time second
time_seconds = 0
set_index = False
items_thread_array = []

response_dict = {}

while(check_dict.stop == False):
    #print(check_dict.stop)


    #Wait 1 seconds if 
    input_q_length = redis.connection.llen(redis.q_name)


    if(input_q_length > 0):

        #Pop the item from the queue
        line = redis.connection.rpop(redis.q_name)

        #GET THE COLLECTION CODE, THEN CREATE THE URL CACHE, AND PASS IT ON

        # tweet_obj = {}
        # ner_obj = {}

        # print(set_index)


        try:
            #Load the full dictionary with different keys inside of it
            data_obj = json.loads(line)

            if(set_index == False):
                if(data_obj.get('aidr') != None):
                    collection_code = data_obj['aidr']['crisis_code']
                    #print("hererer")
                    url_cache.set_index(collection_code + cache_details['name'])
                    
                    image_cache_mapping.create_index_cache(es_object_cache, collection_code + cache_details['name'])
                    image_cache_mapping.create_index_dedup(es_object_dedup, collection_code +"_img_dedup")
                    image_cache_mapping.create_index_image(es_object_cache, collection_code + "_img_index")



                    set_index = True
            # if(data_obj.get('tweet') != None):
            #     tweet_obj = data_obj.get('tweet')
            #     # tweet_obj = json.loads(tweet_data)

            # if(data_obj.get('ner') != None):
            #     ner_obj = data_obj.get('ner')
            #     # ner_obj = json.loads(ner_data)


        except Exception as ex:
            print(ex)
            pass


        items_thread_array.append(threading.Thread(
            target = image_downloader, args = (collection_code, data_obj, url_cache, path_to_download_imgs, response_dict)))
        
    else:
        time.sleep(1)
        time_seconds = time_seconds + 1


    #If batch size is equal mention in config file
    length_check = len(items_thread_array) >= batch_size
    #or there are some items in the input_q and 5 time_seconds have passed 
    time_check = len(items_thread_array) > 0 and (time_seconds % 5 == 0) and (time_seconds != 0)

    if(length_check or time_check):

        for z in range(0, len(items_thread_array)):
            items_thread_array[z].start()

        for y in range(0, len(items_thread_array)):
            items_thread_array[y].join()

        #print(response_dict)

        if(response_dict.get('image_paths') != None):
            # print(len(response_dict['image_paths']))
            for c in range(0, len(response_dict['image_paths'])):
                item_to_push = response_dict['image_paths'][c]
                full_tweet = response_dict['tweets'][c]
                #print(item_to_push)
                for j in range(0, len(output_qs)):

                    json_item_q = {}
                    data_json = {}
                    tweet_id_info = {}

                    insert_in_dict(json_item_q, 'collection_code', collection_code)
                    insert_in_dict(json_item_q, 'image_path', item_to_push)

                    insert_in_dict(tweet_id_info, 'tweet_image_id', item_to_push.split('/')[-1].split('.')[0])
                    insert_in_dict(tweet_id_info, 'json', full_tweet)

                    data_json = json.dumps(json_item_q, indent=None, separators=(',',':'))
                    tweet_json = json.dumps(tweet_id_info, indent=None, separators=(',',':'))

                    if("_IMG_TWEET_INFO" not in output_qs[j]):
                        redis.connection.lpush(collection_code + output_qs[j], data_json)
                    else:
                        redis.connection.lpush(collection_code + output_qs[j], tweet_json)




                    #print(json_item_q)
                    #print(data_json)




        items_thread_array = []
        response_array = []

    if(time_check):
        time_seconds = 0

t1.join()






