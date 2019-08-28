import numpy as np

def dna_encoding(self, index):
    dictDNA = {}
    dictDNA['A'] = np.array([1, 0, 0, 0])
    dictDNA['T'] = np.array([0, 1, 0, 0])
    dictDNA['G'] = np.array([0, 0, 1, 0])
    dictDNA['C'] = np.array([0, 0, 0, 1])
    dictDNA['N'] = np.array([1, 1, 1, 1])

    seq_data = self.idx[index][0]
    seq_data = [char for char in seq_data]

    encoding = np.zeros((len(seq_data), 4))

    for i in range(0, len(seq_data)):
        encoding[i, :] = dictDNA.get(seq_data[i].upper())

    encoding = np.transpose(encoding, [1, 0])

    target = np.array(int(self.idx[index][1]))

    return encoding, target