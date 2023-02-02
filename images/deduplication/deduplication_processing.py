"""
Image Models in the Landslide System
=============================

**Author:** `Ferda Ofli`__

"""

import os
import sys
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import redis
from elasticsearch import Elasticsearch
from copy import deepcopy
from elasticsearch import helpers
import time
from time import gmtime, strftime
import time
from modules.common_utils import cacher, redis_connection, yml_reader, latency_cal
import json


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


#Some global variables
use_gpu = "cuda:1" if torch.cuda.is_available() else "cpu"
device = torch.device(use_gpu)












relative_path = sys.argv[1]
file_name = relative_path.split('/')[1].split('.')[0]
config = yml_reader.ConfigReader(relative_path)


threshold = config.loaded_file['threshold']
classification_batch_size = config.loaded_file['thread_size']
insertion_batch_size = config.loaded_file['thread_size']

redis_queue_name = config.loaded_file['redis']['input_q']

image_index = config.loaded_file['es']['name']

deduplication_index = config.loaded_file['cache']['name']


arch_name = config.loaded_file['arch_name']
best_state_path = config.loaded_file['best_state_path']



redis_host = config.loaded_file['redis']['host']
redis_port = config.loaded_file['redis']['port']


es_host = config.loaded_file['es']['host']
es_port = config.loaded_file['es']['port']


class MyDataset(torch.utils.data.Dataset):
    """Dataset class for wrapping images and target labels read from a file

    Arguments:
        A list of file paths
        Preprocessing transforms
    """

    def __init__(self, file_list, transform=None):
        
        self.transform = transform
        self.X = file_list
        
        self.y = [item.split('/')[-1].split('.')[0] for item in file_list]

        # print(self.X)
        # print(self.y)

        self.samples = list(zip(self.X,self.y))
        

    def __getitem__(self, index):
        path, label = self.samples[index]
        f = open(path,'rb')
        img = Image.open(f)
        if img.mode is not 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def use_model_to_process_images(deduplication_model, preprocess, params):

    global use_gpu, device, batch_size, redis_queue_name, es_host, es_port, redis_host, redis_port, image_index, classification_batch_size, insertion_batch_size

    redis_object = connect_redis(str(redis_host), int(redis_port))
    es_object = connect_elasticsearch(es_host, es_port)
    es_object_cache = connect_elasticsearch(config.loaded_file['cache']['host'], config.loaded_file['cache']['port'])


    if(es_object == None):
        return
    latency = latency_cal.CalculateLatency(insertion_batch_size)


    print(latency)
    bulk_array = []
    results = [] 
    full_path_array = []
    # item_array_dedup = []

    # time.sleep(30)

    counter_file = open("counter_belmont.txt", "w+")

    time_counter = 0

    # counter_index = 26016
    counter_index =0


    while(True):

        total_items_in_redis = redis_object.llen(redis_queue_name)

        #print(total_items_in_redis)
    


        if(total_items_in_redis == 0):
            time.sleep(1)
            time_counter = time_counter + 1
        #print("yess")

        # if(total_items_in_redis > 0  and len(full_path_array) <= 2*insertion_batch_size):
        if(total_items_in_redis > 0):

            #print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

            item = redis_object.rpop(redis_queue_name)
            data_obj = json.loads(item)

            collection_code = data_obj["collection_code"]
            path = data_obj["image_path"]

            full_path_array.append(str(path))
            #img_ids_array.append(str(image_id))

            #print(len(full_path_array))
            if(len(full_path_array) >= classification_batch_size and len(full_path_array) >0):

                latency.start_time()

                img_dataset = MyDataset(full_path_array, preprocess)
                # img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=classification_batch_size, shuffle=False, num_workers=10)
                img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=classification_batch_size, shuffle=False)


                with torch.no_grad():
                    for idx, (inputs, image_ids) in enumerate(img_loader):
                        inputs = inputs.to(device)
                        outputs = deduplication_model(inputs)
                        dense_vector_array = outputs.squeeze(0).cpu().numpy().tolist()

                        results.extend(list(zip(dense_vector_array,list(image_ids))))

                # print(results)


                        #print(type(dense_vector_array[0]))
            #print(len(dense_vector))


                for i in range(0, len(results)):

                    #print(results[i])
                    
                    (dense_vector_array_item, id_image) = results[i]

                    #print(len(dense_vector_array_item))

                    # count_res = es_object.count(body=None, index=deduplication_index)
                    # print(count_res)
                    ref = es_object_cache.indices.refresh(index=deduplication_index)

                    QUERY = '{"track_total_hits":true,"size":1,"_source":["image_id"],"sort":[{"_score":"asc"}],"query":{"script_score":{"query":{"match_all":{}},"script":{"source":"l2norm(params.query_vector, \'dense_vector\')","params":{"query_vector":' + str(dense_vector_array_item) +'}}}}}'
                    
                    #print(QUERY)
                    response = es_object_cache.search(index=deduplication_index, body=QUERY)


                    #print(response['hits']['hits'][0]['_score'])
                    # if(len(response['hits']['hits']) == 0):
                    #     doc = {}
                    #     insert_in_dict(doc, 'counter', 0)
                    #     insert_in_dict(doc, 'dense_vector', dense_vector_array_item)
                    #     insert_in_dict(doc, 'image_id', id_image)
                    #     res = es_object_cache.index(index=deduplication_index, id=int(counter_index%100000), document=doc)
                    # else:
                    try:
                        #print(response['hits'])
                        if(response['hits']['hits'][0]['_score'] >= threshold):


                            doc = {}

                            insert_in_dict(doc, 'counter', int(counter_index%100000))

                            insert_in_dict(doc, 'dense_vector', dense_vector_array_item)
                            insert_in_dict(doc, 'image_id', id_image)

                            # res = es_object.index(index=deduplication_index, id=id_image, document=doc)
                            # item_array_dedup.append(
                            # {
                            # "_index": deduplication_index, 
                            # "_id": int(counter_index%100000),
                            # "_op_type": "update", 
                            # "doc": deepcopy(doc),
                            # "doc_as_upsert": True}
                            # )

                            # res = helpers.bulk(es_object, item_array_dedup)
                            #print(doc)
                            # try:
                            # res = es_object_cache.index(index=deduplication_index, id=int(counter_index%100000), document=doc)
                            # except:
                            res = es_object_cache.index(index=deduplication_index, id=int(counter_index%100000), body=doc)
                            # time.sleep(2)
                            #ref = es_object_cache.refresh(index=deduplication_index)

                            # item_array_dedup = []
                            counter_index = counter_index + 1
                            counter_file.write(str(counter_index%100000)+'\n')
                            counter_file.flush()
    			#print(res)


                            doc_info = {}
                            insert_in_dict(doc_info, 'dedup', {})

                            insert_in_dict(doc_info['dedup'], 'duplicate', False)

                            try:
                                insert_in_dict(doc_info['dedup'], 'duplicate_score', float(response['hits']['hits'][0]['_score']))
                            except:
                                insert_in_dict(doc_info['dedup'], 'duplicate_score', 0)
                                pass

                              # #image id or false
                              # "duplicate": {
                              #   "type": "boolean"
                              # },
                              # "duplicate_image_id": {
                              #   "type": "keyword"
                              # },
                              # "duplicate_score": {
                              #   "type": "float"
                              # },
                            bulk_array.append(
                                {
                                "_index": image_index, 
                                "_id": id_image,
                                "_op_type": "update", 
                                "doc": deepcopy(doc_info),
                                "doc_as_upsert": True,
                                "retry_on_conflict": 5}
                                )


                        else:
                            doc_info = {}
                            insert_in_dict(doc_info, 'dedup', {})

                            insert_in_dict(doc_info['dedup'], 'duplicate', True)
                            insert_in_dict(doc_info['dedup'], 'duplicate_image_id', response['hits']['hits'][0]['_source']['image_id']) 


                            try:
                                insert_in_dict(doc_info['dedup'], 'duplicate_score', float(response['hits']['hits'][0]['_score']))
                            except:
                                insert_in_dict(doc_info['dedup'], 'duplicate_score', 0)
                                pass


                            #print(img_ids_array[i])
                              # #image id or false
                              # "duplicate": {
                              #   "type": "keyword"
                              # },
                              # "duplicate_score": {
                              #   "type": "float"
                              # },

                            bulk_array.append(
                                {
                                "_index": image_index, 
                                "_id": id_image,
                                "_op_type": "update", 
                                "doc": deepcopy(doc_info),
                                "doc_as_upsert": True,
                                "retry_on_conflict": 5}
                                )
                    except:

                        doc = {}

                        insert_in_dict(doc, 'counter', int(counter_index%100000))

                        insert_in_dict(doc, 'dense_vector', dense_vector_array_item)
                        insert_in_dict(doc, 'image_id', id_image)

                        # res = es_object.index(index=deduplication_index, id=id_image, document=doc)
                        # item_array_dedup.append(
                        # {
                        # "_index": deduplication_index, 
                        # "_id": int(counter_index%100000),
                        # "_op_type": "update", 
                        # "doc": deepcopy(doc),
                        # "doc_as_upsert": True}
                        # )

                        # res = helpers.bulk(es_object, item_array_dedup)
                        # try:
                        #     res = es_object_cache.index(index=deduplication_index, id=int(counter_index%100000), document=doc)
                        # except:
                        res = es_object_cache.index(index=deduplication_index, id=int(counter_index%100000), body=doc)

                results = []
                full_path_array = []


        if(len(bulk_array) >= insertion_batch_size and len(bulk_array) > 0):
            try:
                res = helpers.bulk(es_object, bulk_array)
                latency.end_time()
                latency.get_latency_throughput()
                insertion_batch_size = insertion_batch_size*2
                classification_batch_size = classification_batch_size*2
                latency.batch_size = classification_batch_size
                #print(bulk_array)

                #print(total_items_processed)
                #print(res)
                bulk_array = []

                #print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

            except Exception as ex:
                print(ex)
                print("Bulk insertion issue...")


# deduplication
params_duplicate = {
        'arch_name': arch_name,
        'batch_size': classification_batch_size,
        'best_state_path': best_state_path,
        'num_classes': 365,
        'img_resize': 256,
        'input_size': 224
        }

model_duplicate = torchvision.models.__dict__[params_duplicate['arch_name']](num_classes=params_duplicate['num_classes'])
params_duplicate['checkpoint'] = torch.load(params_duplicate['best_state_path'], map_location=device)
state_dict = {str.replace(k,'module.',''): v for k,v in params_duplicate['checkpoint']['state_dict'].items()}
model_duplicate.load_state_dict(state_dict)
model_duplicate.fc = Identity()
for p in model_duplicate.parameters():
    p.requires_grad = False
model_duplicate.to(device)
model_duplicate.eval()
print(' deduplication model loaded.')

preprocess_duplicate = torchvision.transforms.Compose([
torchvision.transforms.Resize((params_duplicate['input_size'],params_duplicate['input_size'])),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


use_model_to_process_images(model_duplicate, preprocess_duplicate, params_duplicate)





