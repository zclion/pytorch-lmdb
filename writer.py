from PIL import Image
import numpy as np
import datum_pb2
import lmdb


def array_to_datum(arr, label):
    assert arr.ndim == 3
    assert arr.dtype == np.uint8
    assert label is not None

    datum = datum_pb2.Datum()
    datum.width, datum.height, datum.channels = arr.shape
    datum.data = arr.tostring()
    datum.label = label
    return datum


def preprocess(img):
    # TODO put your code here
    return np.asarray(img, dtype=np.uint8)


def save_to_lmdb(save_path, imgs):
    """
    :param save_path: lmdb path(dir, not file)
    :param imgs: img path and label list
    """
    db = lmdb.open(save_path, map_size=1024 ** 4)
    txn = db.begin(write=True)

    count = 0
    for img_path in imgs:
        # TODO put your code here
        split = img_path.split()
        assert len(split) == 2
        img_path = split[0]
        label = int(split[1])
        img = Image.open(img_path).convert('RGB')
        img = preprocess(img)
        datum_img = array_to_datum(img, label)
        txn.put('{:0>8d}'.format(count).encode(), datum_img.SerializeToString())

        count += 1
        if count % 1000 == 0:
            print('processed %d images' % count)

    print('num_samples: ', count)
    txn.put('num_samples'.encode(), str(count).encode())
    txn.commit()
    db.close()
