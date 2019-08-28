import torch
from torchvision import datasets, transforms
from utils.args import get_args

class Splice(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.idx = self.generate_idx(dir)

    def generate_idx(self, dir):
        filecsv = dir+'spliceTrain.csv'
        ids = []
        with open(filecsv) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                ids.append(row)
        return ids

    def __getitem__(self, index):

        encoding,target = dna_encoding(index)

        encoding = encoding.astype(np.float32)
        target = target.astype(np.float32)

        encoding = torch.from_numpy(encoding)
        target = torch.from_numpy(target)
        target = target.type(torch.LongTensor)

        return encoding,target