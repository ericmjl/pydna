from pydna.parsers import parse, parse_primers
from pydna.amplify import pcr

primers = """
>3CYC1clon
CGAUGTCGACTTAGATCTCACAGGCTTTTTTCAAG

>5CYC1clone
GAUCGGCCGGATCCAAATGACTGAATTCAAGGCC
"""
template = """
>templ
CgATGTCGACTTAGATCTCACAGGCTTTTTTCAAGaCGGCCTTGAATTCAGTCATTTGGATCCGGCCGAtC
"""
fp, rp = parse_primers(primers)
te, *rest = parse(template)
te.add_feature()
p = pcr((fp, rp, te))
print(p.figure())

hej = p.seq

from Bio.SeqFeature import SeqFeature

"""
5CgATGTCGACTTAGATCTCACAGGCTTTTTTCAAG...GGCCTTGAATTCAGTCATTTGGATCCGGCCGAtC3
                                       ||||||||||||||||||||||||||||||||||
                                      3CCGGAACTTAAGTCAGTAAACCTAGGCCGGCUAG5
5CGAUGTCGACTTAGATCTCACAGGCTTTTTTCAAG3
 |||||||||||||||||||||||||||||||||||
3GcTACAGCTGAATCTAGAGTGTCCGAAAAAAGTTC...CCGGAACTTAAGTCAGTAAACCTAGGCCGGCTaG5
"""

import re

wpos = [0] + [m.start() for m in re.finditer('U', hej.watson)] + [len(hej.watson)]
cpos = [0] + [m.start() for m in re.finditer('U', hej.crick)] + [len(hej.crick)]

from itertools import tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return list(slice(x, y, 1) for x, y in zip(a, b))


wslices = pairwise(wpos)
cslices = pairwise(cpos)

ln = len(hej)
for ws in wslices:
    for cs in cslices:
        if ws.stop >= ln - cs.stop:
            pass
            # print(ws, cs)
        print(ws, cs)
