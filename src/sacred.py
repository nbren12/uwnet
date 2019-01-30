"""Functions for getting stuff from sacred"""
import pymongo
from bson.objectid import ObjectId
import pandas as pd
from toolz import valmap
import re
import gridfs

database = 'uwnet'


def find_id(coll, id):
    return coll.find_one({"_id": ObjectId(id)})


def get_grid_file(id):
    db = get_database()
    fs = gridfs.GridFS(db)
    return fs.get(id)


def get_database():
    client = pymongo.MongoClient()
    return client[database]


def get_metrics_collection():
    return get_database().metrics


def get_run(id):
    return get_database()['runs'].find_one({'_id': id})


def get_last_model_id(id):
    run_data = get_run(id)

    model_files = []
    for artifact in run_data['artifacts']:
        match = re.search(r"(\d+)\.pkl", artifact['name'])
        if match:
            model_files.append((int(match.group(1)), artifact))

    return [x[1] for x in sorted(model_files, key=lambda x: x[0])][-1]


def get_last_model(id):
    artifact = get_last_model_id(id)
    return get_grid_file(artifact['file_id'])


def get_metrics(id):
    doc = get_run(id)
    metrics = doc['info']['metrics']

    output = {}
    for metric in metrics:
        metricid = metric['id']
        name = metric['name']
        output[name] = find_id(get_metrics_collection(), metricid)

    return output


def convert_metric_to_dataframe(metric):
    return pd.DataFrame({
        metric['name'] : metric['values'],
        'timestamps': metric['timestamps'],
        'steps': metric['steps'],
    })


def get_metrics_as_pandas(id):
    return valmap(convert_metric_to_dataframe, get_metrics(id))
