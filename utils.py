# -*- coding: utf-8 -*-

import pathlib
import pickle

basedir = pathlib.os.path.abspath(pathlib.os.path.dirname(__file__))

__all__ = ['to_pathlib', 'unpacking', 'packing']

def to_pathlib(path: str) -> pathlib.Path:
    """
    A function to convert path in to Pathlib.Path

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    pathlib.Path
        Pothlib.Path generated after appending basedir to input file path.

    """
    path = pathlib.Path(pathlib.os.sep.join(path)).as_posix()
    path = pathlib.os.path.join(basedir, path)
    return pathlib.Path(path).absolute()


def unpacking(pickle_file_path: bytes) -> object:
    print("unpacking ...")
    with open(pickle_file_path, 'rb') as f:
        pickled_object = f.read()  
    unpacked_object = pickle.loads(pickled_object)
    del pickled_object
    return unpacked_object


def packing(source_object: object, pickle_file_path: str) -> bytes:
    print("packing ...")
    if source_object is None:
        return
    pickled_object = pickle.dumps(source_object)
    with open(pickle_file_path, 'wb') as f:
        f.write(pickled_object)
    return pickled_object