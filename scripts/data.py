# Based on https://github.com/facebookresearch/CPC_audio/blob/zerospeech/cpc/dataset.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Originally released under the MIT license.
import numpy as np
from pathlib import Path
from random import shuffle
import time
import torch
from torch.multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
import tqdm


def extractShape(path):
    seq = loadFile(path)
    return seq.shape


def loadFilePool(path):
    seqName = path.stem
    seq = loadFile(path)
    return seqName, seq


def loadFile(path):
    ext = Path(path).suffix
    if ext == '.txt':
        data = np.loadtxt(path)
    elif ext == '.npy':
        data = np.load(path)
    elif ext == '.pt':
        data = torch.load(path)
    else:
        raise ValueError(f'Format not supported ({ext}).')
    return data


class SequentialData(Dataset):
    def __init__(self,
                 path,
                 seqNames,
                 nProcessLoader=50,
                 MAX_SIZE_LOADED=4000000000):
        """
        Args:
            - path (string): path to the training dataset
            - seqNames (list): sequences to load
            - nProcessLoader (int): number of processes to call when loading the
                                    data from the disk
            - MAX_SIZE_LOADED (int): target maximal size of the floating array
                                     containing all loaded data.
        """
        self.MAX_SIZE_LOADED = MAX_SIZE_LOADED
        self.nProcessLoader = nProcessLoader
        self.dbPath = Path(path)
        self.seqNames = [self.dbPath / x for _, x in seqNames]
        self.reload_pool = Pool(nProcessLoader)

        self.prepare()
        self.data = []

        self.loadNextPack(first=True)
        self.loadNextPack()

    def getSeqNames(self):
        return [str(x) for x in self.seqNames]

    def clear(self):
        if 'data' in self.__dict__:
            del self.data

    def prepare(self):
        shuffle(self.seqNames)
        start_time = time.time()

        print("Checking length...")
        allShape = self.reload_pool.map(extractShape, self.seqNames)

        self.packageIndex, self.totSize = [], 0
        start, packageSize = 0, 0
        for index, shape in tqdm.tqdm(enumerate(allShape)):
            packageSize += shape[0]
            if packageSize * shape[1] > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, index])
                self.totSize += packageSize
                start, packageSize = index, 0

        if packageSize > 0:
            self.packageIndex.append([start, len(self.seqNames)])
            self.totSize += packageSize

        print(f"Done, elapsed: {time.time() - start_time:.3f} seconds")
        print(f'Scanned {len(self.seqNames)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f"{len(self.packageIndex)} chunks computed")
        self.currentPack = -1
        self.nextPack = 0

    def getNPacks(self):
        return len(self.packageIndex)

    def loadNextPack(self, first=False):
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            start_time = time.time()
            print('Joining pool')
            self.r.wait()
            print(f'Joined process, elapsed={time.time()-start_time:.3f} secs')
            self.nextData = self.r.get()
            self.parseNextDataBlock()
            del self.nextData
        self.nextPack = (self.currentPack + 1) % len(self.packageIndex)
        seqStart, seqEnd = self.packageIndex[self.nextPack]
        #if self.nextPack == 0 and len(self.packageIndex) > 1:
        #    self.prepare()
        self.r = self.reload_pool.map_async(loadFilePool,
                                            self.seqNames[seqStart:seqEnd])

    def parseNextDataBlock(self):
        # To accelerate the process a bit
        self.nextData.sort(key=lambda x: (x[0], x[1]))
        tmpData = []

        for seqName, seq in self.nextData:
            tmpData.append(seq)
            del seq

        self.data = torch.cat(tmpData, dim=0)

    def __len__(self):
        return self.totSize

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            print(idx)
        outData = self.data[idx].view(1, -1)
        return outData

    def getNLoadsPerEpoch(self):
        return len(self.packageIndex)

    def getDataLoader(self, batchSize, numWorkers=0):
        r"""
        Get a batch sampler for the current dataset.
        Args:
            - batchSize (int): batch size
        """
        nLoops = len(self.packageIndex)
        totSize = self.totSize // batchSize

        def samplerCall():
            sampler = UniformAudioSampler(len(self.data))
            return BatchSampler(sampler, batchSize, True)

        return SequentialLoader(self, samplerCall, nLoops, self.loadNextPack, totSize,
                                numWorkers)


class SequentialLoader(object):
    r"""
    A DataLoader meant to handle an SequentialData object.
    In order to handle big datasets SequuentialData works with big chunks of
    data it loads sequentially in memory: once all batches have been sampled
    on a chunk, the SequentialData loads the next one.
    """
    def __init__(self,
                 dataset,
                 samplerCall,
                 nLoop,
                 updateCall,
                 size,
                 numWorkers):
        r"""
        Args:
            - dataset (AudioBatchData): target dataset
            - samplerCall (function): batch-sampler to call
            - nLoop (int): number of chunks to load
            - updateCall (function): function loading the next chunk
            - size (int): total number of batches
            - numWorkers (int): see torch.utils.data.DataLoader
        """
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers

    def __len__(self):
        return self.size

    def __iter__(self):

        for i in range(self.nLoop):
            sampler = self.samplerCall()
            dataloader = DataLoader(self.dataset,
                                    batch_sampler=sampler,
                                    num_workers=self.numWorkers)
            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()


class UniformAudioSampler(Sampler):

    def __init__(self, dataSize):
        self.len = dataSize

    def __iter__(self):
        return iter(torch.randperm(self.len).tolist())

    def __len__(self):
        return self.len
