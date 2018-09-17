"""Logging object used by training routine"""
import os

import attr

import torch
from datetime import datetime


def get_git_rev():
    import subprocess
    import uwnet
    root_dir = os.path.dirname(uwnet.__path__[0])
    out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root_dir)
    return out.decode().strip()


class DBLogger(object):
    def log_run(self, args, config):
        self.run_id = self.insert_run({
            'start_timestamp': datetime.utcnow(),
            'training_data': os.path.abspath(args.input),
            "config": config,
            "args": vars(args),
            'git': {
                'rev': get_git_rev()
            }
        })  # yapf: disable

    def log_batch(self, batch_info):
        batch_info['run_id'] = self.run_id
        self.insert_batch(batch_info)

    def log_epoch(self, i, lstm):
        path = f"{i}.pkl"
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, path)
        self.insert_epoch({
            'epoch': i,
            'run_id': self.run_id,
            'model': os.path.abspath(path)
        })


@attr.s
class TinyDBLogger(DBLogger):
    """Class for logging data to a TinyDB"""
    path = attr.ib()

    @property
    def db(self):
        import tinydb
        return tinydb.TinyDB(self.path)

    # epoch_table = db.table('epochs')

    def insert_run(self, arg):
        run_table = self.db.table('runs')
        return run_table.insert(arg)

    def insert_batch(self, arg):
        table = self.db.table('batches')
        return table.insert(arg)

    def insert_epoch(self, arg):
        table = self.db.table('epochs')
        return table.insert(arg)


@attr.s
class MongoDBLogger(DBLogger):
    database = attr.ib(default='uwnet')

    @property
    def db(self):
        from pymongo import MongoClient
        return MongoClient()[self.database]

    def insert(self, collection, post):
        return self.db[collection].insert_one(post).inserted_id

    def insert_batch(self, post):
        """can use meta class for this, but not worth it"""
        return self.insert('batch', post)

    def insert_epoch(self, post):
        return self.insert('epoch', post)

    def insert_run(self, post):
        return self.insert('run', post)
