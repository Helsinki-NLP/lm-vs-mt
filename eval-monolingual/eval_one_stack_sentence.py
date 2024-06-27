import argparse
import collections
import itertools
import multiprocessing as mp
import pathlib
import os
import random
import numpy as np

import fairseq
from fairseq import hub_utils
import sentencepiece as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
import tqdm
import youtokentome as yttm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.autograd.set_detect_anomaly(True)

def set_seed(seed_value):
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # PyTorch to ensure reproducibility.
    torch.cuda.manual_seed(seed_value)  # for GPU
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # To further ensure reproducibility
    torch.backends.cudnn.benchmark = False  # Not using the inbuilt cudnn auto-tuner.


class PredictionModel(nn.Module):
	""" a straightforward classifier head chucked onto a pretrained model. """

	def __init__(self, fairseq_model_path: str, n_classes: int, n_inputs: int=1, sentence_level: bool=True, fine_tuning: bool=False, extracted_layer: int=-1):
		if n_inputs > 1:
			assert sentence_level
		super().__init__()
		loaded = hub_utils.from_pretrained(fairseq_model_path, checkpoint_file='checkpoint_best.pt')
		# loaded['models'][0].decoder.layers = loaded['models'][0].decoder.layers[:extracted_layer]
		self.model = loaded['models'][0]
		self.task = loaded['task']
		self.fs_args = loaded['args']
		self.n_inputs = n_inputs
		self.n_classes = n_classes
		self.head = nn.Sequential(
			nn.Dropout(p=0.1),
			nn.Linear(model_d * n_inputs, hidden_d),
			nn.GELU(),
			nn.Linear(hidden_d, n_classes),
		)
		self.sentence_level = sentence_level
		self.fine_tuning = fine_tuning
		self.extracted_layer = extracted_layer
		if not self.fine_tuning:
			self.model.eval()
			for p in self.model.parameters():
				p.requires_grad = False
		else:
			self.model.train()
		for p in self.head.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
			else:
				nn.init.zeros_(p)

	def train(self, mode=True):
		super().train(mode)
		if mode and not self.fine_tuning:
			self.model.eval()

	def forward(self, *inputs):
		with torch.set_grad_enabled(self.fine_tuning):
			all_features = []
			for ipt in inputs:
				state, features_dict = self.model.extract_features(ipt, encoder_out=None)
				all_features.append(features_dict['inner_states'][self.extracted_layer].transpose(0,1))
		if self.sentence_level:
			all_features = [f.sum(1) for f in all_features]
		features = torch.cat(all_features, dim=-1)
		preds = self.head(features)
		return preds


def to_tokens(sent):
	pass

class TokenLevelDataset(data.Dataset):
	def __init__(self, file, labels_dict=None, add_post=True, add_pre=False):
		self._items = []
		ntoks = 0
		if labels_dict is None:
			labels_dict = collections.defaultdict(itertools.count().__next__)
		with open(file) as istr:
			def iter_sentences():
				accum = []
				for line in map(str.strip, istr):
					if line:
						accum.append(line.split())
					else:
						if accum: yield accum
						accum = []
				if accum: yield accum
			for sentence in iter_sentences():
				tokens, labels = zip(*sentence)
				sent = ' '.join(tokens)
				true_tok_idx = 0
				spm_labels = []
				spm_tokens = []
				for tok, label in zip(tokens, labels):
					pieces = to_tokens(tok)
					spm_tokens.extend(pieces)
					spm_labels.extend([labels_dict[label]] + [-100] * (len(pieces) - 1))
				assert len(spm_labels) == len(spm_tokens)
				if spm_tokens != to_tokens(sent):
					assert len(spm_tokens) == len(to_tokens(sent))
					tqdm.tqdm.write(f'[Warning] mismatch in token-wise/sentence-wise segmentations:\n\ttokens: {tokens}\n\ttoken-wise: {spm_tokens},\n\tsentence-wise: {to_tokens(sent)}')
				if add_pre:
					spm_labels = [-100] + spm_labels
				if add_post:
					spm_labels = spm_labels + [-100]
				self._items.append({
					'source': [' '.join(spm_tokens)],
					'target': spm_labels,
				})
				ntoks += len(tokens)
			self.labels_dict = dict(labels_dict)
			self.ntoks = ntoks

	def __getitem__(self, idx):
		return self._items[idx]

	def __len__(self):
		return len(self._items)


class SentenceLevelDataset(data.Dataset):
	def __init__(self, file, labels_dict=None, **unused):
		self._items = []
		if labels_dict is None:
			labels_dict = collections.defaultdict(itertools.count().__next__)
		with open(file) as istr:
			# header = next(istr)
			for sentence in map(str.strip, istr):
				*sents, label = sentence.split('\t')
				spm_sents = [' '.join(to_tokens(sent)) for sent in sents]
				self._items.append({
					'source': spm_sents,
					'target': labels_dict[label],
				})
			self.labels_dict = dict(labels_dict)

	def __getitem__(self, idx):
		return self._items[idx]

	def __len__(self):
		return len(self._items)


parser = argparse.ArgumentParser()
parser.add_argument('modeldir', type=str)
parser.add_argument('--trainfile', type=str)
parser.add_argument('--validfiles', type=str, nargs='*', default=[])
parser.add_argument('--testfiles', type=str, nargs='*', default=[])
parser.add_argument('--n_inputs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--extracted_layer', type=int, default=-1)
parser.add_argument('--grad_accum', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--sentence_level', action='store_true')
parser.add_argument('--fine_tuning', action='store_true')
parser.add_argument('--spm_pretokenization', action='store_true')
parser.add_argument('--has_bos', action='store_true')
parser.add_argument('--has_eos', action='store_true')
parser.add_argument('--save_path', type=str, default='./', help='Path to save the trained model')
parser.add_argument('--seed_value', type=int, default=42)
parser.add_argument('--model_d', type=int, default=512)
parser.add_argument('--hidden_d', type=int, default=256)
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
args = parser.parse_args()

set_seed(args.seed_value)
print(args.seed_value)

model_d = args.model_d
hidden_d = args.hidden_d

if not args.spm_pretokenization:
	spm = yttm.BPE(model='/scratch/project_2005099/data/lm-vs-mt/duplicate-prob-eval/yttm-bpe-vocab/lm-vs-mt.bpe.model')
	outtype = yttm.OutputType.SUBWORD
	def to_tokens(sent):
		return spm.encode([sent], output_type=outtype)[0]
else:
	spm = sp.SentencePieceProcessor('/scratch/project_2005099/data/lm-vs-mt/processed/spm/mt-vs-lm.spm.model')
	outtype = str
	def to_tokens(sent):
		return spm.encode(sent, out_type=outtype)


DatasetClass = {
	True: SentenceLevelDataset,
	False: TokenLevelDataset,
}[args.sentence_level]


train_dataset = DatasetClass(args.trainfile, add_pre=args.has_bos, add_post=args.has_eos)

valid_datasets = [
	DatasetClass(
		validfile,
		labels_dict=train_dataset.labels_dict,
		add_pre=args.has_bos,
		add_post=args.has_eos,
	)
	for validfile in args.validfiles
]

model = PredictionModel(
	args.modeldir,
	n_classes=len(train_dataset.labels_dict),
	n_inputs=args.n_inputs,
	sentence_level=args.sentence_level,
	fine_tuning=args.fine_tuning,
	extracted_layer=args.extracted_layer,
).to(args.device)
first_fake_idx = next(
	itertools.chain(
		(idx for idx, symb in enumerate(model.task.source_dictionary.symbols) if symb.startswith('madeupword')),
		[len(model.task.source_dictionary)]
	)
)
unk_idx =  model.task.source_dictionary.unk_index

def add_ctrl_tokens(sent):
	if args.has_bos:
		sent = f'{model.task.source_dictionary.bos_word.strip()} {sent.strip()}'
	if args.has_eos:
		sent = f'{sent.strip()} {model.task.source_dictionary.eos_word.strip()}'
	return sent.strip()

def collate_fn(items):
	sources = zip(*[it['source'] for it in items])
	sources = [
		[model.task.source_dictionary.encode_line(add_ctrl_tokens(s), append_eos=False, add_if_not_exist=False)[...,:128] for s in src]
		for src in sources
	]
	if args.sentence_level:
		targets = torch.tensor([it['target'] for it in items])
	else:

		targets = [torch.tensor(it['target']) for it in items]
		targets = pad_sequence(targets, padding_value=-100, batch_first=True)
	batch = {
		'source': [
			pad_sequence(src, padding_value=model.task.source_dictionary.pad_index, batch_first=True)
			for src in sources
		],
		'target': targets
	}
	warnable = [(src == unk_idx).sum().item() for src in batch['source']]
	if any(s > 5 for s in warnable):
		tqdm.tqdm.write('[warning, more than 5 unks in batch]')
	#	import pdb; pdb.set_trace()
	# batch['source'] = [src.masked_fill(src >= first_fake_idx, unk_idx) for src in batch['source']]
	return batch


train_loader = data.DataLoader(
	train_dataset,
	batch_size=args.batch_size,
	shuffle=True,
	collate_fn=collate_fn,
	num_workers=8,
	prefetch_factor=40,
	persistent_workers=True,
)

valid_dataloaders = [
	data.DataLoader(
		valid_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collate_fn,
		num_workers=8,
		prefetch_factor=40,
		persistent_workers=True,
	)
	for valid_dataset in valid_datasets
]


#optimizer = fairseq.optim.adafactor.Adafactor(
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=0.0001)
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
val_criterion = nn.CrossEntropyLoss(reduction='sum')

best_validloss = float('inf')
best_epoch = 0
from sklearn.metrics import f1_score
acc = collections.deque(maxlen=256)

print('n classes:', train_dataset.labels_dict)
optimizer.zero_grad()
total_steps = 0
for epoch in tqdm.trange(args.epochs, desc='epochs'):
	model.train()
	pbar = tqdm.tqdm(train_loader, desc=f'train e{epoch}')
	#tqdm.tqdm.write(f'sample params: {model.model.decoder.layers[0].self_attn.v_proj.bias[:10].tolist()}')
	for batch in pbar:
		sources = [t.to(args.device) for t in batch['source']]
		target = batch['target'].to(args.device)
		preds = model(*sources)
		loss = criterion(preds.view(target.numel(), -1), target.view(-1))
		loss.backward()
		acc.append((preds.view(target.numel(), -1).argmax(-1) == target.view(-1)).float().mean().item())
		pbar.set_postfix({'L': loss.item(), 'A': sum(acc)/len(acc)})
		total_steps += 1
		if total_steps % args.grad_accum == 0:
			optimizer.step()
			optimizer.zero_grad()
	total_loss = 0
	with torch.no_grad():
		model.eval()
		for valid_file, valid_dataloader in tqdm.tqdm(zip(args.validfiles, valid_dataloaders), total=len(args.validfiles), desc='validations', leave=False):
			this_loss = 0.0
			this_accuracy = 0.0
			all_preds, all_targets = [], []
			for batch in tqdm.tqdm(valid_dataloader, desc=f'valid {valid_file} e{epoch}', leave=False):
				sources = [t.to(args.device) for t in batch['source']]
				target = batch['target'].to(args.device)
				preds = model(*sources)
				loss = val_criterion(preds.view(target.numel(), -1), target.view(-1))
				this_loss += loss
				top1_preds = preds.view(target.numel(), -1).argmax(dim=-1)
				targets = target.view(-1)
				this_accuracy += (top1_preds == targets).masked_select(targets != -100).float().sum().item()
				all_preds.extend(top1_preds.masked_select(targets != -100).tolist())
				all_targets.extend(targets.masked_select(targets != -100).tolist())
			if args.sentence_level:
				denom = len(valid_dataloader.dataset)
			else:
				denom = valid_dataloader.dataset.ntoks
			this_loss /= denom
			this_accuracy /= denom
			tqdm.tqdm.write(f'valid {valid_file} e{epoch}: L={this_loss:.4f}, Acc={this_accuracy:.4f}')
			tqdm.tqdm.write(f'\tF1 macro={f1_score(all_targets, all_preds, average="macro"):.4f}, micro={f1_score(all_targets, all_preds, average="micro"):.4f}, baseline={1/len(set(all_targets)):.4f}')
			total_loss += this_loss
		tqdm.tqdm.write(f'total valid e{epoch}: L={total_loss:.4f}')
		if total_loss < best_validloss:
			tqdm.tqdm.write(f'\tbested previous optimal loss ({best_validloss:.4f}), saving model.')
			best_validloss = total_loss

			# save model to specific path
			save_path = os.path.join(args.save_path, 'best.pt')
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			torch.save(model.state_dict(), save_path)

			# torch.save(model.state_dict(), 'best.pt')
			best_epoch = epoch
	if epoch > (best_epoch + args.patience):
		break
	else:
		tqdm.tqdm.write(f'best epoch: {best_epoch}, current epoch: {epoch}, patience: {args.patience + best_epoch - epoch} / {args.patience}')

# model.load_state_dict(torch.load('best.pt'))
model.load_state_dict(torch.load(save_path))
model = model.to(args.device)
test_dataloaders = [
	data.DataLoader(
		DatasetClass(
			testfile,
			labels_dict=train_dataset.labels_dict,
			add_pre=args.has_bos,
			add_post=args.has_eos,
		),
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collate_fn,
		num_workers=mp.cpu_count(),
		prefetch_factor=40 // mp.cpu_count(),
		persistent_workers=True,
	)
	for testfile in args.testfiles
]

with torch.no_grad():
	model.eval()
	for testfile, testloader in zip(args.testfiles, test_dataloaders):
		this_loss = 0
		this_accuracy = 0
		all_preds, all_targets = [], []
		for batch in tqdm.tqdm(testloader, desc=f'test {testfile}', leave=False):
			sources = [t.to(args.device) for t in batch['source']]
			target = batch['target'].to(args.device)
			preds = model(*sources)
			loss = val_criterion(preds.view(target.numel(), -1), target.view(-1))
			this_loss += loss
			top1_preds = preds.view(target.numel(), -1).argmax(dim=-1)
			targets = target.view(-1)
			this_accuracy += (top1_preds == targets).masked_select(targets != -100).float().sum().item()
			all_preds.extend(top1_preds.masked_select(targets != -100).tolist())
			all_targets.extend(targets.masked_select(targets != -100).tolist())
		if args.sentence_level:
			denom = len(testloader.dataset)
		else:
			denom = testloader.dataset.ntoks
		this_loss /= denom
		this_accuracy /= denom
		test_output = f'Test {testfile}: L={this_loss:.4f}, Acc={this_accuracy:.4f}\n'
		tqdm.tqdm.write(test_output)
		test_output += f'\tF1 macro={f1_score(all_targets, all_preds, average="macro"):.4f}, micro={f1_score(all_targets, all_preds, average="micro"):.4f}, baseline={1/len(set(all_targets)):.4f}\n'
		tqdm.tqdm.write(test_output)
		
		output_text_path = os.path.join(args.save_path, 'test_results.txt')
		with open(output_text_path, 'w') as output_file:
			output_file.write(test_output)