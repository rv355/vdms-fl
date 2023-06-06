import vdms
import requests
from models import CNNCifar
import numpy as np

NODES = []
WORKER_NODES = []


def initialize_worker_nodes(nrof_nodes):
    for i in range(nrof_nodes):
        NODES.append("http://localhost:"+str(5010+i))

def selection_score(constraints, options):
    # selection_query = {
    #     "FindImage" : {
    #         "constraints": constraints,
    #         "operations": [
    #             {
    #                 "type": "userOp",
    #                 "id": "selection_score",
    #                 "options": {
    #                     "selection_score_model": options["selection_score_model"]                        
    #                 }
    #             }
    #         ]
    #     }
    # }

    selection_query = {
        "FindImage" : {
            "constraints": constraints,
            "results": {
                "list": ["label"]
            }
        }
    }

    for node in NODES:
        response = requests.post(node+"/selection", json=selection_query)
        ss_value = response.json()["selection_score"]
        print(node, ss_value)
        if ss_value >= options["selection_score_threshold"]:
            WORKER_NODES.append(node)

def training(constraints, options):

    model = CNNCifar(10)

    model.to('cpu')
    model.train()
    print(model)

    weights = model.state_dict()

    train_query = {
        "FindImage" : {
            "constraints": constraints,
            "operations" : [
                {
                    "type": "grouped_userOp",
                    "id": "sampling",
                    "options": {
                        "batch_size": options["batch_size"] if "batch_size" in options.keys() else 10,
                        "epochs": options["epoch"] if "epoch" in options.keys() else 10
                    }
                },
                {
                    "type": "resize",
                    "width": options["width"],
                    "height": options["height"],
                },
                {
                    "type": "userOp",
                    "id": "preprocess",
                    "options": {
                        "pre_processing_steps": options["pre_processing_steps"]
                    }
                },
                {
                    "type": "grouped_userOp",
                    "id": "fltrain",
                    "options": {
                        "global_model": model,
                        "model": options["model"],
                        "epoch": options["epoch"],
                        "validation_size": options["validation_size"],
                        "learing_rate": options["learning_rate"]
                    }
                }
            ]
        }
    }

    response = requests.post(NODES[0]+"/selection", data=train_query)

def testing():
    return

def aggregate():
    return

def fedlearn(query):
    # selection_score(query["FindImage"]["constraints"], query["FindImage"]["operations"][0]["options"])

    print(WORKER_NODES)

    training(query["FindImage"]["constraints"], query["FindImage"]["operations"][0]["options"])

if __name__ == '__main__':

    initialize_worker_nodes(1)

    query = {
        "FindImage" : {
            "constraints": {
                "dataset": ["==", "cifar10"],
            },
            "operations": [
                {
                    "type": "userOp",
                    "id": "fedlearn",
                    "options": {                        
                        "model": "CNNCifar",
                        "max_rounds": 100,
                        "accuracy": 94,                                                
                        "selection_score_model": "nse",
                        "selection_score_threshold": 0.7,
                        "width": 300,
                        "height": 200,
                        "pre_processing_steps": ["graying, dilate"],
                        "epoch": 100,
                        "validation_size": 0.2,
                        "learning_rate": 0.0001,
                        "aggregation_model": "FedAvg"
                    }
                }
            ]
        }
    }

    fedlearn(query)