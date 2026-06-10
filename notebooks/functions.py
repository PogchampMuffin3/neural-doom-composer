from torch.utils.data import Dataset, DataLoader

CONTEXT = 512
STRIDE  = 256
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

    def __getitem__(self, idx):
        si, start = self.index[idx]
        chunk = self.seqs[si][start: start+self.context+1]
        if self.augment:
            allowed = self._allowed_offsets(chunk)
            off = random.choice(allowed)
            if off != 0:
                chunk = SHIGF_MAPS[off][chunk]
                
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y