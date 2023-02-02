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
from modules.common_utils import cacher, redis_connection, yml_reader, latency_cal
import json

def initialize_model(params):

    # Extract necessary variables
    arch_name = params['arch_name']
    num_classes = len(params['checkpoint']['class_names'])
    use_pretrained = False
    
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model = None
    input_size = 0
    img_resize = 0

    if arch_name == "resnet18":
        """ Resnet18
        """
        model = torchvision.models.resnet18(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "resnet50":
        """ Resnet50
        """
        model = torchvision.models.resnet50(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "resnet101":
        """ Resnet101
        """
        model = torchvision.models.resnet101(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "alexnet":
        """ Alexnet
        """
        model = torchvision.models.alexnet(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "vgg":
        """ VGG11_bn
        """
        model = torchvision.models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "vgg16":
        """ VGG16_bn
        """
        model = torchvision.models.vgg16_bn(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "squeezenet":
        """ Squeezenet
        """
        model = torchvision.models.squeezenet1_0(pretrained=use_pretrained)
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224
        img_resize = 256

    elif arch_name == "densenet":
        """ Densenet
        """
        model = torchvision.models.densenet121(pretrained=use_pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "mobilenet":
        """ MobileNet
        """
        model = torchvision.models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = model.last_channel
        model.fc = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "efficientnet":
        """ EfficientNet
        """
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
        input_size = 224
        img_resize = 256

    elif arch_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = torchvision.models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 299
        img_resize = 360

    else:
        print("Invalid model name, exiting...",flush=True)
        exit()

    return model, img_resize, input_size


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

best_state_path = config.loaded_file['best_state_path']

arch_name = config.loaded_file['arch_name']

classification_batch_size = batch_size

insertion_batch_size = batch_size

# redis_queue_name = 'emsc_live_landslides_index_relevancy'

# image_index = 'emsc_live_landslides_index'





# es_host = '10.4.2.107'
# es_port = 9500

# redis_host = '10.4.2.95'
# redis_port = 6379




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

def convert_label(label):
    # print(label)
    if label == 'not_relevant':
        return False
    else:
        return True


def use_model_to_process_images(relevancy_model, preprocess, params):

    global use_gpu, device, batch_size, redis_queue_name, es_host, es_port, redis_host, redis_port, image_index, classification_batch_size, insertion_batch_size

    redis_object = connect_redis(config.loaded_file['redis']['host'], int(config.loaded_file['redis']['port']))
    es_object = connect_elasticsearch(config.loaded_file['es']['host'], int(config.loaded_file['es']['port']))

    if(es_object == None):
        return

    bulk_array = []

    full_path_array = []
    img_ids_array = []
    results = []


    time_counter = 0

    # time.sleep(60)
    image_index = config.loaded_file['es']['name']
    redis_queue_name = config.loaded_file['redis']['input_q']

    latency = latency_cal.CalculateLatency(batch_size)


    print(latency)

    while(True):
        #while(insertion_batch_size <= 5000):

        total_items_in_redis = redis_object.llen(redis_queue_name)

    


        # if(total_items_in_redis == 0):
        #     time.sleep(30)
        #     time_counter = time_counter + 1


        if(total_items_in_redis > 0  and len(full_path_array) <= 2*insertion_batch_size):
            
            start = time.time()
            item = redis_object.rpop(redis_queue_name)

            data_obj = json.loads(item)

            collection_code = data_obj["collection_code"]
            path = data_obj["image_path"]

            image_id = path.split('/')[-1].split('.')[0]

            full_path_array.append(str(path))
            img_ids_array.append(str(image_id))

            #print(data_obj)





        if(len(full_path_array) >= 1):
            if(len(full_path_array) >= insertion_batch_size):
                
                #print(img_ids_array)
                #print("here")

                #classification part
                latency.start_time()

                img_dataset = None
                img_loader = None

                img_dataset = MyDataset(full_path_array, preprocess)
                # img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=classification_batch_size, shuffle=False)
                img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=10, shuffle=False, num_workers=10)
    

                with torch.no_grad():


                    for idx, (inputs, image_ids) in enumerate(img_loader):



                        inputs = inputs.to(device)
                        outputs = relevancy_model(inputs)
                        outputs = torch.nn.functional.softmax(outputs,1)
                        probs, preds = torch.max(outputs, 1)
                        #print(relevancy_model)
                        preds_ = [params['checkpoint']['class_names'][p] for p in preds.cpu().numpy()]
                        probs_ = probs.cpu().numpy().tolist()
                        
                        results.extend(list(zip(preds_,probs_,list(image_ids))))

                latency.end_time()

                #print(results)
                for i in range(0, len(results)):
                    doc = {}
                    (label, prob, id_image) = results[i]


                    insert_in_dict(doc, 'relevancy', {})

                    insert_in_dict(doc['relevancy'], 'relevant', convert_label(label))
                    insert_in_dict(doc['relevancy'], 'relevant_conf', prob)

                    #print(img_ids_array[i])

                    bulk_array.append(
                        {
                        "_index": image_index, 
                        "_id": id_image,
                        "_op_type": "update", 
                        "doc": deepcopy(doc),
                        "doc_as_upsert": True,
                        "retry_on_conflict": 5}
                        )

                full_path_array = []
                img_ids_array = []
                results = []


        if(len(bulk_array) >= insertion_batch_size and len(bulk_array) > 0):
            # print("Length of bulk array " +str(len(bulk_array)))
            #try catch else
            try:
                res = helpers.bulk(es_object, bulk_array)
                latency.get_latency_throughput()
                insertion_batch_size = insertion_batch_size*2
                classification_batch_size = classification_batch_size*2
                latency.batch_size = classification_batch_size

                # print(len(bulk_array))
                # print(res)
                #print(total_items_processed)
                #print(res)
                done = time.time()
                elapsed = done - start

                bulk_array = []

                # print("Time elapsed")
                # print(elapsed)
                #insertion_batch_size = insertion_batch_size + 100


            except Exception as ex:
                print(ex)
                print("Bulk insertion issue...")
                pass

params_relevancy = {
        'arch_name': arch_name,
        'batch_size': classification_batch_size,
        'best_state_path': best_state_path
        }

params_relevancy['checkpoint'] = torch.load(params_relevancy['best_state_path'], map_location=device)
model_relevancy, params_relevancy['img_resize'], params_relevancy['input_size'] = initialize_model(params_relevancy)
model_relevancy.load_state_dict(params_relevancy['checkpoint']['state_dict'])
for p in model_relevancy.parameters():
    p.requires_grad = False
    
model_relevancy.to(device)
model_relevancy.eval()
print('relevancy model loaded...')

preprocess_relevancy = torchvision.transforms.Compose([
torchvision.transforms.Resize((params_relevancy['input_size'],params_relevancy['input_size'])),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


use_model_to_process_images(model_relevancy, preprocess_relevancy, params_relevancy)

