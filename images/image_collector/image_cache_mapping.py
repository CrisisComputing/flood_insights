"""
   Created by Umair Qazi -
   Research Assistant at
   Qatar Computing Research Institute
   Jul 15, 2018

   For elasticsearch 7.2
"""

# -*- coding: utf-8 -*-
import json
import csv
import datetime
import random
import sys
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


def create_index_cache(es_object, index_name):
    created = False
    # index settings
    settings = {
      "settings": {
        "index": {
          "number_of_shards": 5,
          "number_of_replicas": 2,
          "refresh_interval": "1s"
        }
      },
      "mappings": {
        "properties": {
          "hash_str": {
            "type": "keyword"
          },
          "url": {
            "type": "keyword"
          },
          "origin_tweet_id": {
            "type": "keyword"
          },
          "image_extension": {
            "type": "keyword"
          },
          "image_id": {
            "type": "keyword"
          },
          "image_name": {
            "type": "keyword"
          },
          "error": {
            "type": "keyword"
          },
          "duplicate_array": {
            "type": "nested",
            "properties": {
              "tweet_id": {
                "type": "keyword"
              }
            }
          },
          "duplicate_count": {
            "type": "integer"
          },
          "download_status": {
            "type": "boolean"
          }
        }
      }
    }



    try:
        if not es_object.indices.exists(index_name):
        # Ignore 400 means to ignore "Index Already Exist" error.
            es_object.indices.create(index=index_name, body=settings)
            print('Created Index: ' + index_name)
    except Exception as ex:
        print(str(ex))
    # finally:
    #     return created


def create_index_dedup(es_object, index_name):
    created = False
    # index settings
    settings = {
      "settings": {
        "index": {
          "number_of_shards": 5,
          "number_of_replicas": 2,
          "refresh_interval": "1s"
        }
      },    "mappings": {
      "properties": {
        "counter": {
          "type": "integer"
        },
        "dense_vector": {
          "type": "dense_vector",
          "dims": 2048
        },
        "image_id": {
          "type": "keyword"
        }
      }
    }
  }
    



    try:
        if not es_object.indices.exists(index_name):
        # Ignore 400 means to ignore "Index Already Exist" error.
            es_object.indices.create(index=index_name, body=settings)
            print('Created Index: ' + index_name)
    except Exception as ex:
        print(str(ex))
    # finally:
    #     return created

def create_index_image(es_object, index_name):
    created = False
    # index settings
    settings = {
      "settings": {
        "index": {
          "number_of_shards": 5,
          "number_of_replicas": 2,
          "refresh_interval": "1s"
        }
      },
      "mappings": {
        "properties": {
          "classifier_label_backup": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "created_at": {
            "type": "keyword"
          },
          "date": {
            "type": "date",
            "format": "strict_date_hour_minute_second"
          },
          "day_date": {
            "type": "date"
          },
          "day_date_strict": {
            "type": "date",
            "format": "strict_date"
          },
          "deleted": {
            "type": "boolean"
          },
          "dedup": {
            "properties": {
              "duplicate": {
                "type": "keyword"
              },
              "duplicate_score": {
                "type": "float"
              }
            }
          },
          "duplicate_image_id": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "human_label": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "image_id": {
            "type": "keyword"
          },
          "incident": {
            "properties": {
              "confidence": {
                "type": "float"
              },
              "label": {
                "type": "keyword"
              }
            }
          },
          "image_path": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "in_reply_to_id": {
            "type": "keyword"
          },
          "landslide": {
            "type": "boolean"
          },
          "landslide_conf": {
            "type": "float"
          },
          "lang": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "location_info": {
            "properties": {
              "coordinates": {
                "properties": {
                  "address": {
                    "properties": {
                      "city": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      },
                      "country": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country_code": {
                        "type": "keyword"
                      },
                      "county": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "neighbourhood": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "state": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      }
                    }
                  },
                  "geohash": {
                    "type": "geo_point"
                  },
                  "lat": {
                    "type": "double"
                  },
                  "lon": {
                    "type": "double"
                  },
                  "query_term": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword"
                      }
                    }
                  },
                  "response": {
                    "type": "text",
                    "index": False
                  }
                }
              },
              "place": {
                "properties": {
                  "address": {
                    "properties": {
                      "city": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      },
                      "country": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country_code": {
                        "type": "keyword"
                      },
                      "county": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "neighbourhood": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "state": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      }
                    }
                  },
                  "geohash": {
                    "type": "geo_point"
                  },
                  "lat": {
                    "type": "double"
                  },
                  "lon": {
                    "type": "double"
                  },
                  "query_term": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword"
                      }
                    }
                  },
                  "response": {
                    "type": "text",
                    "index": False
                  }
                }
              },
              "resolved_location": {
                "properties": {
                  "address": {
                    "properties": {
                      "city": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      },
                      "country": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country_code": {
                        "type": "keyword"
                      },
                      "county": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "neighbourhood": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "state": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      }
                    }
                  },
                  "geohash": {
                    "type": "geo_point"
                  },
                  "lat": {
                    "type": "double"
                  },
                  "lon": {
                    "type": "double"
                  },
                  "query_term": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword"
                      }
                    }
                  },
                  "response": {
                    "type": "text",
                    "index": False
                  }
                }
              },
              "resolved_location_source": {
                "type": "keyword"
              },
              "tweet_text": {
                "type": "nested",
                "properties": {
                  "address": {
                    "properties": {
                      "city": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      },
                      "country": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country_code": {
                        "type": "keyword"
                      },
                      "county": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "neighbourhood": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "state": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      }
                    }
                  },
                  "geohash": {
                    "type": "geo_point"
                  },
                  "lat": {
                    "type": "double"
                  },
                  "lon": {
                    "type": "double"
                  },
                  "query_term": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword"
                      }
                    }
                  },
                  "response": {
                    "type": "text",
                    "index": False
                  }
                }
              },
              "tweet_text_single": {
                "properties": {
                  "address": {
                    "properties": {
                      "city": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country_code": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "county": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "neighbourhood": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "state": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      }
                    }
                  },
                  "geohash": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                      }
                    }
                  },
                  "lat": {
                    "type": "float"
                  },
                  "lon": {
                    "type": "float"
                  },
                  "query_term": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                      }
                    }
                  },
                  "response": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                      }
                    }
                  }
                }
              },
              "user_location": {
                "properties": {
                  "address": {
                    "properties": {
                      "city": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      },
                      "country": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country_code": {
                        "type": "keyword"
                      },
                      "county": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "neighbourhood": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "state": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      }
                    }
                  },
                  "geohash": {
                    "type": "geo_point"
                  },
                  "lat": {
                    "type": "double"
                  },
                  "lon": {
                    "type": "double"
                  },
                  "query_term": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword"
                      }
                    }
                  },
                  "response": {
                    "type": "text",
                    "index": False
                  }
                }
              },
              "user_profile_description_location": {
                "type": "nested",
                "properties": {
                  "address": {
                    "properties": {
                      "city": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      },
                      "country": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "country_code": {
                        "type": "keyword"
                      },
                      "county": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "neighbourhood": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                          }
                        }
                      },
                      "state": {
                        "type": "text",
                        "fields": {
                          "keyword": {
                            "type": "keyword"
                          }
                        }
                      }
                    }
                  },
                  "geohash": {
                    "type": "geo_point"
                  },
                  "lat": {
                    "type": "double"
                  },
                  "lon": {
                    "type": "double"
                  },
                  "query_term": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword"
                      }
                    }
                  },
                  "response": {
                    "type": "text",
                    "index": False
                  }
                }
              }
            }
          },
          "quoted_id": {
            "type": "keyword"
          },
          "relevancy": {
            "properties": {
              "relevant": {
                "type": "boolean"
              },
              "relevant_conf": {
                "type": "float"
              }
            }
          },
          "retweeted_id": {
            "type": "keyword"
          },
          "text": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 512
              }
            }
          },
          "tweet_id": {
            "type": "keyword"
          },
          "user": {
            "dynamic": False,
            "properties": {
              "created_at": {
                "type": "keyword"
              },
              "description": {
                "type": "text",
                "fields": {
                  "keyword": {
                    "type": "keyword"
                  }
                }
              },
              "description_ner": {
                "type": "nested",
                "properties": {
                  "label": {
                    "type": "keyword"
                  },
                  "name": {
                    "type": "keyword"
                  }
                }
              },
              "favourites_count": {
                "type": "integer"
              },
              "followers_count": {
                "type": "integer"
              },
              "friends_count": {
                "type": "integer"
              },
              "gender": {
                "type": "keyword"
              },
              "id": {
                "type": "long"
              },
              "id_str": {
                "type": "keyword"
              },
              "listed_count": {
                "type": "integer"
              },
              "location": {
                "type": "text",
                "fields": {
                  "keyword": {
                    "type": "keyword"
                  }
                }
              },
              "name": {
                "type": "keyword"
              },
              "name_ner": {
                "type": "nested",
                "properties": {
                  "label": {
                    "type": "keyword"
                  },
                  "name": {
                    "type": "keyword"
                  }
                }
              },
              "screen_name": {
                "type": "keyword"
              },
              "statuses_count": {
                "type": "integer"
              },
              "time_zone": {
                "type": "keyword"
              },
              "type": {
                "type": "keyword"
              },
              "verified": {
                "type": "boolean"
              }
            }
          }
        }
      }
    }
    



    try:
        if not es_object.indices.exists(index_name):
        # Ignore 400 means to ignore "Index Already Exist" error.
            es_object.indices.create(index=index_name, body=settings)
            print('Created Index: ' + index_name)
    except Exception as ex:
        print(str(ex))
    # finally:
    #     return created



# es_object_107 = connect_elasticsearch("10.4.2.107", 9500)
# es_object_100 = connect_elasticsearch("10.4.2.100", 9200)

# # # Change the name of the index
# # raw_tweets = "test_es_full_schema"
# create_index_cache(es_object_107, "belmont_image_url_cache")
# create_index_dedup(es_object_100, "belmont_image_dedup")
# create_index_image(es_object_107, "belmont_image_index")



