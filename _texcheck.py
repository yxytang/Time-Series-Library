# -*- coding: utf-8 -*-
"""Static sanity checks on the generated LaTeX."""
import io
import collections

BS = chr(92)          # backslash
s = io.open('PA-STL-DualTimesNet.tex', encoding='utf-8').read()
body = '\n'.join('' if l.strip().startswith('%') else l.split('%')[0]
                 for l in s.split('\n'))

print('--- environment pairing ---')
bad = {}
for key in ('figure', 'table', 'equation', 'abstract', 'document',
            'thebibliography', 'tabular', 'itemize', 'center'):
    b = body.count(BS + 'begin{' + key + '}')
    e = body.count(BS + 'end{' + key + '}')
    if b or e:
        print('  %-16s begin=%d end=%d %s' % (key, b, e, 'OK' if b == e else '** MISMATCH **'))
        if b != e:
            bad[key] = (b, e)

print('--- character balance ---')
print('  braces      :', body.count('{') - body.count('}'))
print('  dollar signs:', body.count('$'), '(must be even)')
print('  \\left/\\right:', body.count(BS + 'left'), body.count(BS + 'right'))

print('--- commands used ---')
for c in ('operatorname', 'underset', 'middle', 'surd', 'includegraphics',
          'keywords', 'articletype', 'affil', 'email', 'title', 'author',
          'section', 'subsection', 'caption', 'label', 'bibitem', 'cite'):
    n = body.count(BS + c)
    if n:
        print('  %-16s %d' % (BS + c, n))

print('--- anything still non-ASCII ---')
print(' ', sorted(set(ch for ch in body if ord(ch) > 127)) or 'none')

print('--- file references that must exist ---')
import os
for m in ('figures/',):
    pass
missing = []
for f in os.listdir('LATEX/figures') if os.path.isdir('LATEX/figures') else []:
    pass
i = 0
while True:
    i = body.find(BS + 'includegraphics', i)
    if i < 0:
        break
    j = body.find('}', i)
    name = body[body.find('{', i) + 1:j]
    if not os.path.exists(os.path.join('LATEX', name)):
        missing.append(name)
    i = j
print('  missing graphics:', missing or 'none')
