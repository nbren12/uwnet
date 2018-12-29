"""Functions for getting stuff from sacred"""
import pymongo
import re
import gridfs

database = 'uwnet'


def get_grid_file(id):
    db = get_database()
    fs = gridfs.GridFS(db)
    return fs.get(id)


def get_database():
    client = pymongo.MongoClient()
    return client[database]


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
