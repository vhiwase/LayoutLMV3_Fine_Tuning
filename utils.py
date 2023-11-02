import pickle

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