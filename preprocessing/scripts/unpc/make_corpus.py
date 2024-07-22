import argparse
import pathlib
import random
random.seed(935324)
import xml.etree.ElementTree as ET

import pandas as pd
import tqdm

def lang2dir(lang):
  return lang, pathlib.Path('UNv1.0-TEI')

parser = argparse.ArgumentParser()
parser.add_argument('src', type=lang2dir)
parser.add_argument('tgt', type=lang2dir)
parser.add_argument('-allocs', default='capped-lp-allocations.txt')
parser.add_argument('-spm_file', required=True)
parser.add_argument('links', type=lambda p: list(pathlib.Path(p).glob('**/*.lnk')))
parser.add_argument('out_trainfile_bitext')
parser.add_argument('out_trainfile_docs')
parser.add_argument('out_valfile_bitext')
parser.add_argument('out_valfile_docs')
args = parser.parse_args()

allocs = pd.read_csv(args.allocs, sep=' ', header=None, names=['src', 'tgt'])
slang, sdir = args.src
tlang, tdir = args.tgt
allocs = allocs[allocs['src'].str.startswith(slang)].reset_index(drop=True)
allocs = allocs[allocs['tgt'].str.startswith(tlang)].reset_index(drop=True)
allocs['true_src'] = allocs['src'].apply(lambda f: sdir / f)
allocs['true_tgt'] = allocs['tgt'].apply(lambda f: tdir / f)

alloc_refs = {k: None for k in allocs.apply(lambda row: (row['src'], row['tgt']), axis=1)}

print('read links...')
with tqdm.trange(len(alloc_refs), desc='relevant <linkGrp>', position=2) as pbar, \
tqdm.tqdm(position=1, desc='all <linkGrp>', unit_scale=True, unit_divisor=1000) as pbar_link_grp, \
tqdm.tqdm(position=0, desc='all elems', unit_scale=True, unit_divisor=1000) as pbar_all_elems:
  for link_file in tqdm.tqdm(args.links, desc='files', position=3):
    with open(link_file, 'r') as istr:
      try:
        for _, elem in ET.iterparse(istr):
          pbar_all_elems.update()
          if elem.tag == 'linkGrp':
            pbar_link_grp.update()
            sdoc, tdoc = elem.attrib.get('fromDoc', -1), elem.attrib.get('toDoc', -1)
            sdoc = sdoc[len('Xml/'):]
            tdoc = tdoc[len('Xml/'):]
            paired_doc = (sdoc, tdoc)
            if paired_doc in alloc_refs:
              assert alloc_refs[paired_doc] is None
              alloc_refs[paired_doc] = elem
              pbar.update()
      except ET.ParseError:
        tqdm.tqdm.write(f'[WARNING] failed to parse {link_file}')

allocs['link_grp'] = allocs.apply(
  lambda row: alloc_refs.get(
    (row['src'], row['tgt']),
     None,
  ),
  axis=1,
)

if allocs['link_grp'].isna().any():
  len_pre = len(allocs)
  allocs = allocs.dropna(subset='link_grp').reset_index()
  len_post = len(allocs)
  print(f'[WARNING] corpus down to {len_post} / {len_pre} ({len_post / len_pre * 100:.2f}%)')


import sentencepiece as sp
sp_model = sp.SentencePieceProcessor(model_file=args.spm_file)

def get_contents(row):
  try:
    with open(row.true_src, 'r') as src_str:
      src = {
        elem.attrib['id']: ' '.join(elem.text.strip().split() if elem.text is not None else [])
        for _, elem in ET.iterparse(src_str)
        if elem.tag == 's'
      }
    with open(row.true_tgt, 'r') as src_str:
      tgt = {
        elem.attrib['id']: ' '.join(elem.text.strip().split() if elem.text is not None else [])
        for _, elem in ET.iterparse(src_str)
        if elem.tag == 's'
      }
    return [
      [
        [' '.join(sp_model.encode(src[sid], out_type=str)) for sid in sids.split()],
        [' '.join(sp_model.encode(tgt[tid], out_type=str)) for tid in tids.split()],
      ]
      for tids, sids in map(
        lambda lnk: lnk.attrib['xtargets'].split(';'),
        row.link_grp.iter(tag='link')
      )
    ]
  except (ET.ParseError, KeyError):
    tqdm.tqdm.write(f'[WARNING] had to drop a pair: {row.true_src} / {row.true_tgt}')
    return None

with open(args.out_trainfile_bitext, 'a') as bitext_train, \
open(args.out_valfile_bitext, 'a') as bitext_val, \
open(args.out_trainfile_docs, 'a') as docs_train,\
open(args.out_valfile_docs, 'a') as docs_val:
  contents_iter = map(get_contents, allocs.itertuples())
  contents_pbar = tqdm.tqdm(contents_iter, desc='write out', total=len(allocs))
  contents_pbar = filter(None, contents_pbar)
  for contents in contents_pbar:
    # write bitexts
    all_src, all_tgt = [], []
    docs_out, bitext_out = (docs_val, bitext_val) if random.uniform(0, 1) <= 1 / 2000 else (docs_train, bitext_train)
    for srcs, tgts in contents:
        if len(srcs) > 0 and len(tgts) > 0:
          print(slang, tlang, ' '.join(srcs), ' '.join(tgts), sep='\t', file=bitext_out)
        all_src.extend(srcs)
        all_tgt.extend(tgts)
    # write source & tgt docs
    for src in all_src:
      print(slang, src, sep='\t', file=docs_out)
    print('', file=docs_out)
    for tgt in all_tgt:
      print(tlang, tgt, sep='\t', file=docs_out)
    print('', file=docs_out)

