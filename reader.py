import torch
import torch.utils.data as data
import numpy as np
import random
import lmdb
import datum_pb2


def datum_to_array(datum):
    return np.fromstring(datum.data, dtype=np.uint8).reshape(datum.width, datum.height, datum.channels)


class LmdbTrainDataset(data.Dataset):
    def __init__(self, lmdb_path):
        print('start to load lmdb train dataset')
        db = lmdb.open(lmdb_path, readonly=True)
        self.txn = db.begin()
        self.datum = datum_pb2.Datum()
        self.num_classes = int(self.txn.get('num_classes'.encode()))
        self.num_samples = int(self.txn.get('num_samples'.encode()))
        print('train dataset size:', self.num_samples, ' ids:', self.num_classes)

    def __getitem__(self, idx):
        value = self.txn.get('{:0>8d}'.format(idx).encode())
        self.datum.ParseFromString(value)
        img = datum_to_array(self.datum)
        label = self.datum.label

        # TODO put your code here
        # demo: random flip
        if random.random() < 0.5:
            img = np.fliplr(img)

        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img.copy())
        return img, label

    def __len__(self):
        return self.num_samples


def read_lmdb_train_dataset(batch_size, lmdb_path):
    train_dataset = LmdbTrainDataset(lmdb_path)
    loader = data.DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=8)
    return loader, train_dataset.num_classes
