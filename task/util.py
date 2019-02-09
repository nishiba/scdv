import os
import pickle
import subprocess

import chardet
import luigi
import luigi.format


def make_target(file_path):
    extension = os.path.splitext(file_path)[1]
    if extension == '.pkl':
        return luigi.LocalTarget(file_path, format=luigi.format.Nop)
    return luigi.LocalTarget(file_path)


def dump(target: luigi.LocalTarget, obj):
    extension = os.path.splitext(target.path)[1]
    with target.open('w') as f:
        if extension == '.pkl':
            f.write(pickle.dumps(obj, protocol=4))
        else:
            f.write(str(obj))


def load(target: luigi.LocalTarget):
    with target.open('r') as f:
        return pickle.load(f)


def tokenize(file_path):
    p = subprocess.run(['mecab', '-Owakati', file_path],
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell=False)
    try:
        lines = p.stdout.decode(chardet.detect(p.stdout)["encoding"])
        return lines.split()
    except:
        return None
