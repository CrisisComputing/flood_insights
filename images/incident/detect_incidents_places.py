import os
import pprint
import glob
import json
from tqdm import tqdm
import cv2
import torch
#import matplotlib.pyplot as plt
#%matplotlib inline
import redis
from elasticsearch import Elasticsearch
from copy import deepcopy
from elasticsearch import helpers
import time
from modules.common_utils import cacher, redis_connection, yml_reader, latency_cal
import sys

from architectures import (
    FilenameDataset,
    get_incidents_model,
    update_incidents_model_with_checkpoint,
    update_incidents_model_to_eval_mode,
    get_predictions_from_model
)
from parser import get_parser, get_postprocessed_args

from utils import get_index_to_incident_mapping, get_index_to_place_mapping


# es_object = mapping_generator.connect_elasticsearch()
def connect_elasticsearch(host, port):
    es = None
    es = Elasticsearch([{'host': host, 'port': port, 'timeout':10000, "maxsize": 500000, 'max_retries':1000, 'retry_on_timeout':True}])
    if es.ping():
        print('Elasticsearch Connected')
    else:
        print('Elasticsearch Could NOT connect!')
    return es

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

def connect_redis(host, port):

    #redis_pipe = None

    redis_pipe = redis.StrictRedis(host=host, port=port, charset="utf-8", decode_responses=True)

    #print(str(redis_pipe))

    print(redis_pipe)

    if redis_pipe:
        print('Redis Connected')
    else:
        print('Redis Could NOT connect!')
    return redis_pipe



# model
CONFIG_FILENAME = "configs/multi_label_final_model"
CHECKPOINT_PATH_FOLDER = "pretrained_weights/"

# call command
# python detect_incidents_places.py --config=configs/multi_label_final_model --checkpoint_path=pretrained_weights/ --mode=test --num_gpus=1 --topk=1 --images_file=example_images_list.txt --images_path=example_images --output_file=example_images_incidents_places_multi-label.tsv


parser = get_parser()
args = parser.parse_args()
args = get_postprocessed_args(args)


relative_path = args.config_file
file_name = relative_path.split('/')[1].split('.')[0]
config = yml_reader.ConfigReader(relative_path)





#connecting to redis and 


batch_size = config.loaded_file['thread_size']


#Creating the cache and passing the config details
# cache_details = config.loaded_file['cache']

# url_cache = cacher.Cacher(cache_details['host'], 
#                     cache_details['port'],
#                     cache_details['name'])


#base_path + collection_code + monthly_path
# path_to_download_imgs = config.loaded_file['path_to_download_imgs']


# output_qs = config.loaded_file['redis']['output_qs']

classification_batch_size = batch_size

insertion_batch_size = batch_size

redis_object = connect_redis(config.loaded_file['redis']['host'], int(config.loaded_file['redis']['port']))
es_object = connect_elasticsearch(config.loaded_file['es']['host'], int(config.loaded_file['es']['port']))

image_index = config.loaded_file['es']['name']
redis_queue_name = config.loaded_file['redis']['input_q']


bulk_array = []

full_path_array = []


bulk_array = []
dataset = []
inference_dict = {}
loader = None

latency = latency_cal.CalculateLatency(batch_size)


print(latency)

# throughput_and_latency = []
# data
#pop data here from the queue
# with open(args.images_file,"r") as f:
#     image_filenames = [l.strip() for l in f.readlines() if l.strip()]

while(True):
    #while(insertion_batch_size <= 5000):


    total_items_in_redis = redis_object.llen(redis_queue_name)


    if(total_items_in_redis > 0  and len(full_path_array) <= 2*insertion_batch_size):
        
        item = redis_object.rpop(redis_queue_name)

        data_obj = json.loads(item)

        collection_code = data_obj["collection_code"]
        path = data_obj["image_path"]

        image_id = path.split('/')[-1].split('.')[0]

        full_path_array.append(str(path))



        # file_name = path.split('/')[-1]
        # file_names_arr.append(file_name)

    if(len(full_path_array) >= 1):
        if(len(full_path_array) >= insertion_batch_size):
            #print(len(full_path_array))
            #print(full_path_array)
            latency.start_time()

            incidents_model = get_incidents_model(args)
            update_incidents_model_with_checkpoint(incidents_model, args)
            update_incidents_model_to_eval_mode(incidents_model)

        #   # Set up the data loader for quickly loading images to run inference with.
            targets = [full_path_array[i] for i in range(len(full_path_array))]
            dataset = FilenameDataset(full_path_array, targets)
            # print(dataset)
            loader = torch.utils.data.DataLoader(dataset,batch_size=10,shuffle=False,num_workers=4)
            # loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

            
            for idx, (batch_input, image_paths) in enumerate(loader):
                # run the model, get the output, set the inference_dict
                output = get_predictions_from_model(
                    args,
                    incidents_model,
                    batch_input,
                    image_paths,
                    get_index_to_incident_mapping(),
                    get_index_to_place_mapping(),
                    inference_dict,
                    topk=args.topk
                )

            # print(output)
            # print(inference_dict)

            latency.end_time()

            



            for image_filename in inference_dict:

                image_id = image_filename.split('/')[-1].split('.')[0]
                # print(image_id)

                incident_label = inference_dict[image_filename]['incidents'][0]
                incident_score = inference_dict[image_filename]['incident_probs'][0]

                place_label = inference_dict[image_filename]['places'][0]
                place_score = inference_dict[image_filename]['place_probs'][0]

                doc = {}
                insert_in_dict(doc, 'incident_place', {})

                insert_in_dict(doc['incident_place'], 'incident_label', incident_label)
                insert_in_dict(doc['incident_place'], 'incident_score', float(incident_score))

                insert_in_dict(doc['incident_place'], 'place_label', place_label)
                insert_in_dict(doc['incident_place'], 'place_score', float(place_score))

                #print(img_ids_array[i])

                bulk_array.append(
                    {
                    "_index": image_index, 
                    "_id": image_id,
                    "_op_type": "update", 
                    "doc": deepcopy(doc),
                    "doc_as_upsert": True,
                    "retry_on_conflict": 5}
                    )

            if(len(bulk_array) >= insertion_batch_size and len(bulk_array) > 0):
                # print("Length of bulk array " +str(len(bulk_array)))
                #try catch else
                try:
                    latency.get_latency_throughput()
                    insertion_batch_size = insertion_batch_size*2
                    classification_batch_size = classification_batch_size*2
                    latency.batch_size = classification_batch_size

                    #res = helpers.bulk(es_object, bulk_array)

                    # print(bulk_array)
                    # print(res)
                    #print(total_items_processed)
                    # print(res)

                    bulk_array = []
                    full_path_array = []
                    dataset = []
                    inference_dict.clear()
                    loader = None




                    # print("Time elapsed")
                    # print(elapsed)
                    #insertion_batch_size = insertion_batch_size + 100


                except Exception as ex:
                    print(ex)
                    print("Bulk insertion issue...")
                    pass










# inference_dict contains numpy arrays which are not JSON serializable,
# either convert them to lists before saving, or change the output format to something else, e.g., tsv

#Save the items in ES index here

# with open(args.output_file, "w") as write_file:
#     for image_filename in inference_dict:
#         out_line = image_filename
#         for i in range(args.topk):
#             out_line += "\t{}\t{}".format(inference_dict[image_filename]['incidents'][i],
#                                           inference_dict[image_filename]['incident_probs'][i])
#             out_line += "\t{}\t{}".format(inference_dict[image_filename]['places'][i],
#                                           inference_dict[image_filename]['place_probs'][i])
#         write_file.write("{}\n".format(out_line))





