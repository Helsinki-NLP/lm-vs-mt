import argparse
import pathlib
import random
random.seed(450983)
import xml.etree.ElementTree as ET

import pandas as pd
import tqdm

def lang2dir(lang):
  if lang == 'zh_cn':
    return lang, pathlib.Path('src') / lang / 'OpenSubtitles' / 'xml'
  return lang, pathlib.Path('src') / lang / 'OpenSubtitles' / 'parsed'

parser = argparse.ArgumentParser()
parser.add_argument('src', type=lang2dir)
parser.add_argument('tgt', type=lang2dir)
parser.add_argument('-allocs', default='capped-lp-allocations.txt')
parser.add_argument('-spm_file', required=True)
parser.add_argument('links')
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
allocs['true_src'] = allocs['src'].apply(lambda f: sdir / f[:-len('.gz')])
allocs['true_tgt'] = allocs['tgt'].apply(lambda f: tdir / f[:-len('.gz')])

alloc_refs = {k: None for k in allocs.apply(lambda row: (row['src'], row['tgt']), axis=1)}

print('read links...')
with open(args.links, 'r') as istr:
  with tqdm.trange(len(alloc_refs), desc='relevant <linkGrp>', position=2) as pbar, \
  tqdm.tqdm(position=1, desc='all <linkGrp>', unit_scale=True, unit_divisor=1000) as pbar_link_grp:
    for _, elem in tqdm.tqdm(ET.iterparse(istr), position=0, desc='all elems', unit_scale=True, unit_divisor=1000):
      if elem.tag == 'linkGrp':
        pbar_link_grp.update()
        sourcedoc, targetdoc = elem.attrib.get('fromDoc', -1), elem.attrib.get('toDoc', -1)
        if sourcedoc.startswith('Xml'):
          sourcedoc = 'unpc/UNv1.0-TEI' + sourcedoc[3:]
        if targetdoc.startswith('Xml'):
          targetdoc = 'unpc/UNv1.0-TEI' + targetdoc[3:]
        paired_doc = sourcedoc, targetdoc
        if paired_doc in alloc_refs:
          assert alloc_refs[paired_doc] is None
          alloc_refs[paired_doc] = elem
          pbar.update()

allocs['link_grp'] = allocs.apply(
  lambda row: alloc_refs.get(
    (row['src'], row['tgt']),
     None,
  ),
  axis=1,
)

assert not allocs['link_grp'].isna().any()

import sentencepiece as sp
sp_model = sp.SentencePieceProcessor(model_file=args.spm_file)

def get_contents(row):
  try:
    with open(row.true_src, 'r') as src_str:
      src = {
        int(elem.attrib['id']): ' '.join(' '.join(w.text.strip().split()) for w in elem.iter(tag='w'))
        for _, elem in ET.iterparse(src_str)
        if elem.tag == 's'
      }
    with open(row.true_tgt, 'r') as src_str:
      tgt = {
        int(elem.attrib['id']): ' '.join(' '.join(w.text.strip().split()) for w in elem.iter(tag='w'))
        for _, elem in ET.iterparse(src_str)
        if elem.tag == 's'
      }
    return [
      [
        [' '.join(sp_model.encode(src[int(sid)], out_type=str)) for sid in sids.split()],
        [' '.join(sp_model.encode(tgt[int(tid)], out_type=str)) for tid in tids.split()],
      ]
      for sids, tids in map(
        lambda lnk: lnk.attrib['xtargets'].split(';'),
        row.link_grp.iter(tag='link')
      )
    ]
  except (ET.ParseError, KeyError):
    tqdm.tqdm.write(f'[WARNING] had to drop a pair: {row.true_src} / {row.true_tgt}')
    return None

with open(args.out_trainfile_bitext, 'a') as bitext_train, \
open(args.out_trainfile_docs, 'a') as docs_train, \
open(args.out_valfile_bitext, 'a') as bitext_val, \
open(args.out_valfile_docs, 'a') as docs_val:
  contents_iter = map(get_contents, allocs.itertuples())
  contents_pbar = tqdm.tqdm(contents_iter, desc='write out', total=len(allocs))
  contents_pbar = filter(None, contents_pbar)
  for contents in contents_pbar:
    # write bitexts
    all_src, all_tgt = [], []
    bitext_out, docs_out = (bitext_val, docs_val) if random.uniform(0, 1) <= 1 / 2000 else (bitext_train, docs_train)
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

