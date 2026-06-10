from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import random
import torch


CONTEXT = 512
STRIDE  = 256
MAX_TRANSPOSE = 6
PROC_DIR = '../data/processed/'

with open(os.path.join(PROC_DIR, 'vocab.json')) as f:
    token2id = json.load(f)
id2token = {y: x for x,y in token2id.items()}
VOCAB_SIZE = len(token2id)

id2pitch = {}
for tok, i in token2id.items():
    if tok.startswith('NOTE_ON_') or tok.startswith('NOTE_OFF_'):
        id2pitch[i] = int(tok.split('_')[-1])
noteon_ids = { int(tok.split('_')[-1]): i for tok, i in token2id.items() if tok.startswith('NOTE_ON') }
noteoff_ids = { int(tok.split('_')[-1]): i for tok, i in token2id.items() if tok.startswith('NOTE_OFF') }

SHIFT_MAPS = {}
for off in range(-MAX_TRANSPOSE, MAX_TRANSPOSE+1):
    m = np.arange(VOCAB_SIZE)
    for pitch, idx in noteon_ids.items():
        np_ = pitch + off
        if 0 <= np_ <= 127:
            m[idx] = noteon_ids[np_]
    for pitch, idx in noteoff_ids.items():
        np_ = pitch + off
        if 0 <= np_ <= 127:
            m[idx] = noteoff_ids[np_]
    SHIFT_MAPS[off] = m

class DoomDataset(Dataset):
    def __init__(self, sequences, context=CONTEXT, stride=STRIDE, augment=True):
        self.context=context
        self.augment=augment
        self.seqs=sequences
        self.index=[]
        for si, s in enumerate(sequences):
            last = len(s) - context - 1
            if last<0: continue
            for start in range(0, last+1, stride):
                self.index.append((si, start))

    def __len__(self):
        return len(self.index)

    def _allowed_offsets(self, chunk):
        pitches = [id2pitch[i] for i in chunk if i in id2pitch]
        if not pitches:
            return list(SHIFT_MAPS.keys())
        lo, hi = min(pitches), max(pitches)
        return [off for off in SHIFT_MAPS if 0 <= lo + off and hi + off <= 127]

    def __getitem__(self, idx):
        si, start = self.index[idx]
        chunk = self.seqs[si][start: start+self.context+1]
        if self.augment:
            allowed = self._allowed_offsets(chunk)
            off = random.choice(allowed)
            if off != 0:
                chunk = SHIFT_MAPS[off][np.array(chunk, dtype=np.int64)]
                
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y