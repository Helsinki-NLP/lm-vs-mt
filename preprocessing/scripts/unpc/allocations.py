import collections
import random

random.seed(12345)

allocs = collections.defaultdict(list)
already_used = set()

print('initializing uniquely paired docs')
with open('base-alloc-from-links', 'r') as istr:
	paired_docs = map(str.split, map(str.strip, istr))
	for A, B in paired_docs:
		lang_a = A.split('/')[0]
		lang_b = B.split('/')[0]
		lang_pair = tuple(sorted([lang_a, lang_b]))
		allocs[lang_pair].append((A, B))
		already_used.add(A)
		already_used.add(B)
print('unique docs allocations:')
for lang_pair in allocs:
	print(*lang_pair, len(allocs[lang_pair]))


print('reading unallocated docs')
unallocated_files = collections.defaultdict(list)
with open('paired-docs-with-links', 'r') as istr:
	paired_docs = map(str.split, map(str.strip, istr))
	for A, B in paired_docs:
		lang_a = A.split('/')[0]
		lang_b = B.split('/')[0]
		lang_pair = tuple(sorted([lang_a, lang_b]))
		if A not in already_used and B not in already_used:
			unallocated_files[lang_pair].append((A, B))

print('pre-sort')
for lang_pair in unallocated_files:
	print(lang_pair, end='... ', flush=True)
	unallocated_files[lang_pair].sort()
print('done.')

print('greedy allocation start')
min_idx=0
while True:
	if min_idx >= len(allocs): break
	lang_pair = sorted(allocs, key=lambda lp: len(allocs[lp]))[min_idx]
	if len(unallocated_files[lang_pair]) == 0: min_idx +=1
	else:
		A, B = unallocated_files[lang_pair].pop()
		if A not in already_used and B not in already_used:
			already_used.add(A)
			already_used.add(B)
			allocs[lang_pair].append((A, B))


print('greedy allocation results:')
for k, v in allocs.items():
	print(*k, len(v))

print('capping down & saving')
cap = min(map(len, allocs.values()))
capped_allocs = {lp: random.sample(allocs[lp], cap) for lp in allocs}

with open('capped-lp-allocations.txt', 'w') as ostr:
	for lp in sorted(capped_allocs):
		for A, B in  random.sample(capped_allocs[lp], len(capped_allocs[lp])):
			print(A, B, file=ostr)
with open('uncapped-lp-allocations.txt', 'w') as ostr:
	for lp in sorted(allocs):
		for A, B in  random.sample(allocs[lp], len(allocs[lp])):
			print(A, B, file=ostr)
