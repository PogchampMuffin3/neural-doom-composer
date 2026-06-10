from torch.utils.data import Dataset
import os
import json
import numpy as np
import random
import torch
import pretty_midi

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

class DoomDataset(Dataset):
    def __init__(self, sequences, context, stride, augment, max_transpose):
        self.context=context
        self.augment=augment
        self.seqs=sequences
        self.index=[]
        for si, s in enumerate(sequences):
            last = len(s) - context - 1
            if last<0: continue
            for start in range(0, last+1, stride):
                self.index.append((si, start))

        self.SHIFT_MAPS = {}
        for off in range(-max_transpose, max_transpose+1):
            m = np.arange(VOCAB_SIZE)
            for pitch, idx in noteon_ids.items():
                np_ = pitch + off
                if 0 <= np_ <= 127:
                    m[idx] = noteon_ids[np_]
            for pitch, idx in noteoff_ids.items():
                np_ = pitch + off
                if 0 <= np_ <= 127:
                    m[idx] = noteoff_ids[np_]
            self.SHIFT_MAPS[off] = m

    def __len__(self):
        return len(self.index)

    def _allowed_offsets(self, chunk):
        pitches = [id2pitch[i] for i in chunk if i in id2pitch]
        if not pitches:
            return list(self.SHIFT_MAPS.keys())
        lo, hi = min(pitches), max(pitches)
        return [off for off in self.SHIFT_MAPS if 0 <= lo + off and hi + off <= 127]

    def __getitem__(self, idx):
        si, start = self.index[idx]
        chunk = self.seqs[si][start: start+self.context+1]
        if self.augment:
            allowed = self._allowed_offsets(chunk)
            off = random.choice(allowed)
            if off != 0:
                chunk = self.SHIFT_MAPS[off][np.array(chunk, dtype=np.int64)]
                
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y




def events_to_midi(events, steps_per_beat, bpm=95):
    sec_per_step = (60.0 / bpm) / steps_per_beat
    midi      = pretty_midi.PrettyMIDI()
    insts     = {}     # program -> Instrument (melodyczne)
    drum_inst = None   # jeden wspólny kanał perkusji
    active    = {}     # pitch -> [(start_step, program)]
    cur, last_prog = 0, 0

    for ev in events:
        if ev in ('BOS', 'EOS', 'PAD'):
            continue
        if ev.startswith('SHIFT_'):
            cur += int(ev.split('_')[1])
        elif ev.startswith('PROGRAM_'):
            last_prog = int(ev.split('_')[1])
        elif ev.startswith('NOTE_ON_'):
            pitch = int(ev.split('_')[-1])
            active.setdefault(pitch, []).append((cur, last_prog))
        elif ev.startswith('NOTE_OFF_'):
            pitch = int(ev.split('_')[-1])
            if active.get(pitch):
                start_step, prog = active[pitch].pop(0)
                start = start_step * sec_per_step
                end   = cur * sec_per_step
                if end > start:
                    if prog not in insts:
                        insts[prog] = pretty_midi.Instrument(program=prog)
                    insts[prog].notes.append(pretty_midi.Note(100, pitch, start, end))
        elif ev.startswith('DRUM_'):
            pitch = int(ev.split('_')[1])
            start = cur * sec_per_step
            end   = start + sec_per_step
            if drum_inst is None:
                drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
            drum_inst.notes.append(pretty_midi.Note(100, pitch, start, end))

    for inst in insts.values():
        midi.instruments.append(inst)
    if drum_inst is not None:
        midi.instruments.append(drum_inst)
    return midi