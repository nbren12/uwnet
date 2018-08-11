# coding: utf-8
import pandas as pd
from tinydb import Query, TinyDB


def get_loss_curve(id):
    db = TinyDB("runs.json"); batches = db.table('batches')
    Batch = Query()
    losses = [{key: b[key] for key in ['epoch', 'batch', 'loss']}
            for b in batches.search(Batch.run == id)]

    return pd.DataFrame(losses)

