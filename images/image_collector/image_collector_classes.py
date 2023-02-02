# -*- coding: utf-8 -*- 
from geolib import geohash as lib_geohash
from modules.common_utils.helpers import insert_in_dict
from modules.common_utils import cacher, helpers
from ast import literal_eval
import urllib.request
from copy import deepcopy
from datetime import datetime
import os


class TweetMediaExtractor:
    # def __init__(self, tweet_object, path_to_download_imgs, es_connection, collection_code):
    def __init__(self, collection_code, tweet_object, url_cache, path_to_download_imgs):

        # self.collection_code = collection_code
        self.tweet_object = tweet_object
        self.images = []
        self.tweets = []
        self.path_to_download_imgs = path_to_download_imgs
        self.collection_code = collection_code
        self.tweet_id = tweet_object["id_str"]
        self.media_dict = {}

        self.entities_parser(tweet_object)

        self.download_images(url_cache)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


    def create_monthly_image_folder(self):
        currentMonth = datetime.now().month
        currentYear = datetime.now().year

        # print(str(currentMonth).zfill(2))
        # print(str(currentYear))

        monthly_folder_path = self.path_to_download_imgs+ '/' + str(currentYear)+'-'+str(currentMonth).zfill(2)+"_" +self.collection_code

        isExist = os.path.exists(monthly_folder_path)
        if not isExist:

           # Create a new directory because it does not exist
            try:
               os.makedirs(monthly_folder_path)

               self.path_to_download_imgs = monthly_folder_path
               print("The new directory is created!")
            except Exception as ex:
                print(ex)


        self.path_to_download_imgs = monthly_folder_path



    def my_url(self, u):
        url = None
        if u is not None:
            url = u.split('?')[0]
        return url

    def get_video_url(self, variants):
        br = -1
        for v in variants:
            cbr = v.get('bitrate',-1)
            if cbr > br:
                url = self.my_url(v.get('url',None))
                br = cbr
        return url

    def extract_media(self, extended_entities, media_dict):
        #case media
        if(extended_entities.get("media") != None):

            media_nested_ob = []
            media_array = extended_entities["media"]
            for i in range(0, len(media_array)):
                if(media_array[i]["type"] == 'video'):
                    # print(media_array[i])
                    

                    #go for highest bitrate
                    #extension from media type in video)
                    #video_url = media_array[i]['video_info']['variants'][0]['url']
                    video_url = self.get_video_url(media_array[i]['video_info']['variants'])
                    media_nested_ob.append({"media_url":media_array[i]["media_url_https"],"type":media_array[i]["type"], "extension":video_url.split(".")[-1], 'video_url':video_url})

                    #APPEND THE THUMBNAIL FOR THE VIDEO
                    media_nested_ob.append({"media_url":media_array[i]["media_url_https"],"type":"photo", "extension":media_array[i]["media_url"].split(".")[-1]})




                elif(media_array[i]["type"] == 'animated_gif'):
                    #go for highest bitrate
                    #extension from media type in gif)

                    #gif_url = media_array[i]['video_info']['variants'][0]['url']
                    gif_url = self.get_video_url(media_array[i]['video_info']['variants'])
                    media_nested_ob.append({"media_url":media_array[i]["media_url_https"],"type":media_array[i]["type"], "extension":gif_url.split(".")[-1], 'gif_url':gif_url})

                else:
                    #photo case
                    media_nested_ob.append({"media_url":media_array[i]["media_url_https"],"type":media_array[i]["type"], "extension":media_array[i]["media_url"].split(".")[-1]})




            #print(media_nested_ob)
            if(len(media_nested_ob) > 0):
                insert_in_dict(self.media_dict, 'media', deepcopy(media_nested_ob))


    def entities_parser(self, tweet_object):
        if(tweet_object.get("retweeted_status") != None):

            if(tweet_object["retweeted_status"].get("extended_tweet") != None):

                if(tweet_object["retweeted_status"]["extended_tweet"].get("extended_entities") != None and tweet_object["retweeted_status"]["extended_tweet"].get("entities") != None):
                    self.extract_media(tweet_object["retweeted_status"]["extended_tweet"]["extended_entities"], self.media_dict)

                elif(tweet_object["retweeted_status"]["extended_tweet"].get("entities") != None):
                    self.extract_media(tweet_object["retweeted_status"]["extended_tweet"]["entities"], self.media_dict)

            else:

                if(tweet_object["retweeted_status"].get("extended_entities") != None and tweet_object["retweeted_status"].get("entities") != None):
                    self.extract_media(tweet_object["retweeted_status"]["extended_entities"],  self.media_dict)

                elif(tweet_object["retweeted_status"].get("entities") != None):
                    self.extract_media(tweet_object["retweeted_status"]["entities"], self.media_dict)



        else:
            if(tweet_object.get("extended_tweet") != None):


                if(tweet_object["extended_tweet"].get("extended_entities") != None and tweet_object["extended_tweet"].get("entities") != None):
                    self.extract_media(tweet_object["extended_tweet"]["extended_entities"],  self.media_dict)

                elif(tweet_object["extended_tweet"].get("entities") != None):
                    self.extract_media(tweet_object["extended_tweet"]["entities"],  self.media_dict)

            else:
                if(tweet_object.get("extended_entities") != None and tweet_object.get("entities") != None):
                    self.extract_media(tweet_object["extended_entities"],  self.media_dict)

                elif(tweet_object.get("entities") != None):
                    self.extract_media(tweet_object["entities"], self.media_dict)

    def download_images(self, url_cache):

        self.create_monthly_image_folder()


        image_id = 0
        if(self.media_dict != {}):
            for i in range(0, len(self.media_dict["media"])):
                # print()
                if(self.media_dict["media"][i]["type"] == 'photo'):

                    image_name = self.media_dict["media"][i]["media_url"].split('/')[-1].split('.')[0]

                    # id_term = image_name + self.tweet_id

                    image_url = self.media_dict["media"][i]["media_url"]

                    # print(url_cache.check_item(id_term))


                    if(url_cache.check_item(image_url) == False):



                        #print(self.media_dict["media"][i])
                        #print(image_name)

                        image_id_name = self.tweet_id+'_'+str(image_id)


                        image_name_with_path = self.path_to_download_imgs + "/" +image_id_name+'.'+self.media_dict["media"][i]["extension"]
                        # print(self.media_dict["media"][i]["extension"])

                        try:

                            #'https://pbs.twimg.com/ext_tw_video_thumb/1346928794456776705/pu/img/98jYf_7AprI99faj.jpg'


                            urllib.request.urlretrieve(image_url, image_name_with_path)

                            self.images.append(image_name_with_path)
                            self.tweets.append(self.tweet_object)
                            url_cache.insert_item_image_url_cache(image_url, image_url, self.tweet_id, image_name, True, self.media_dict["media"][i]["extension"], error=None, image_id=image_id_name)
                            image_id = image_id + 1



                        except Exception as ex:

                            url_cache.insert_item_image_url_cache(image_url, image_url, self.tweet_id, image_name, False, self.media_dict["media"][i]["extension"], error=str(ex),  image_id=None)

                            # self.image_url_cache(image_url, self.tweet_id, image_name, False)
                            print(ex)
                            # print(self.tweet_id)
                            pass

                    else:
                        url_cache.update_duplicate_image_url_cache(image_url, self.tweet_id)

                        # url_cache.insert_item_image_url_cache(image_url, image_url, self.tweet_id, image_name, True, image_id_name)


# class CSVImageUrlExtractor:
#     #All attributes of CSVImages Types
#     def __init__():

      

#     def __str__(self):
#         return str(self.__class__) + ": " + str(self.__dict__)



# class TSVImageUrlExtractor:
#     #All attributes of TSVImages Types
#     def __init__():

      

#     def __str__(self):
#         return str(self.__class__) + ": " + str(self.__dict__)


# class ImageDownloader:
#     def __init__(self, url, path_to_save):
#         self.url = url
#         self.path_to_save = path_to_save

#     def __str__(self):
#         return str(self.__class__) + ": " + str(self.__dict__)








