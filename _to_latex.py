# -*- coding: utf-8 -*-
"""Convert the English edition to LaTeX using the IOP template."""
import io
import re

from docx import Document
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = 'PA-STL-DualTimesNet_EN.docx'
OUT = 'PA-STL-DualTimesNet.tex'

M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

EQ = {
 1: r"x_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t",
 2: r"\text{Trend}_t = \frac{1}{k}\sum_{i=-\lfloor k/2\rfloor}^{\lfloor k/2\rfloor} x_{t+i}",
 3: r"\text{Seasonal}_t = \frac{1}{n_c}\sum_{j=0}^{n_c-1}\big(x_{t+jp} - \text{Trend}_{t+jp}\big)",
 4: r"A = \text{Avg}\Big(\text{Amp}\big(\text{FFT}(X_{1D})\big)\Big)",
 5: r"\{f_1,\ldots,f_k\} = \arg\operatorname{Top}_k(A)",
 6: r"p_i = \frac{T}{f_i}",
 7: r"X_{2D,i} = \text{Reshape}_{p_i,\,f_i}\big(\text{Padding}(X_{1D})\big)",
 8: r"X^{l} = \text{TimesBlock}(X^{l-1}) + X^{l-1}",
 9: r"X^{l} = \sum_{i=1}^{k}\hat{A}_{f_i}\cdot\text{Inception}\big(X^{l}_{2D,i}\big)",
 10: r"\mathcal{R}_{\mathcal{XX}}(\tau) = \lim_{L\to\infty}\frac{1}{L}\sum_{t=1}^{L}\mathcal{X}_t\,\mathcal{X}_{t-\tau}",
 11: r"\mathcal{S}_{\mathcal{XX}}(f) = \mathcal{F}(\mathcal{X}_t)\,\overline{\mathcal{F}(\mathcal{X}_t)}",
 12: r"\mathcal{R}_{\mathcal{XX}}(\tau) = \mathcal{F}^{-1}\big(\mathcal{S}_{\mathcal{XX}}(f)\big)",
 13: r"\tau_1,\ldots,\tau_k = \arg\underset{\tau\in\{1,\ldots,L\}}{\operatorname{Top}_k}\big(\mathcal{R}_{\mathcal{QK}}(\tau)\big)",
 14: r"\hat{\mathcal{R}}_{\mathcal{QK}}(\tau_1),\ldots,\hat{\mathcal{R}}_{\mathcal{QK}}(\tau_k) = \text{Softmax}\big(\mathcal{R}_{\mathcal{QK}}(\tau_1),\ldots,\mathcal{R}_{\mathcal{QK}}(\tau_k)\big)",
 15: r"\text{AutoCorrelation}(\mathcal{Q},\mathcal{K},\mathcal{V}) = \sum_{i=1}^{k}\text{Roll}(\mathcal{V},\tau_i)\,\hat{\mathcal{R}}_{\mathcal{QK}}(\tau_i)",
 16: r"Z = \text{LayerNorm}\big(X + \text{AutoCorrelation}(X)\big)",
 17: r"H = \text{LayerNorm}\big(Z + \text{FFN}(Z)\big)",
 18: r"\hat{X}[j] = \sum_{t=0}^{T-1} X[t]\,e^{-2\pi ijt/T}",
 19: r"\mathcal{P}_{\text{FFT}} = \left\{\frac{T}{j}\ \middle|\ j = 1,\ldots,\left\lfloor \tfrac{T}{2}\right\rfloor\right\}",
 20: r"\tilde{X}[k] = \sum_{t=0}^{T-1} X[t]\,\cos\!\left(\frac{\pi k(2t+1)}{2T}\right)",
 21: r"H = \text{LayerNorm}\Big(x + \mathbf{W}_f\big[\text{Conv}(x);\text{AutoCorr}(x)\big] + \mathbf{b}_f\Big)",
 22: r"\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|",
 23: r"\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}",
 24: r"\text{MASE} = \frac{1}{N}\sum_{i=1}^{N}\frac{|y_i-\hat{y}_i|}{\frac{1}{N-96}\sum_{j=97}^{N}|y_j-y_{j-96}|}",
 25: r"R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{N}(y_i-\bar{y})^2}",
}

FIGS = ['figA_stl.png', 'fig2_timesnet.png', 'fig3_autoformer_enc.png',
        'fig4_architecture.png', 'figB_mic.png', 'fig8_loss.png',
        'fig5_model_bars.png', 'fig6_pred_curves.png', 'fig7_scatter.png',
        'figC_boxplot.png', 'fig4_ablation_bars.png']

# substitutes applied *after* the LaTeX-special characters are escaped
# Tables whose one-line header is wider than the 153 mm text block: the header
# is split over two rows and the body typeset a size smaller so the table fits.
WIDE = {
    7: (['Dataset', 'Month (season)', 'Volatility', 'TimesNet', 'Proposed', 'MAE'],
        ['', '', '', 'MAE/RMSE', 'MAE/RMSE', 'improvement']),
}

SYM = [('±', r'$\pm$'), ('≥', r'$\geq$'), ('≤', r'$\leq$'), ('×', r'$\times$'),
       ('²', r'$^2$'), ('−', '-'), ('–', '--'), ('—', '---'),
       ('“', '``'), ('”', "''"), ('‘', '`'), ('’', "'"),
       ('°', r'$^\circ$'), ('·', r'$\cdot$'), ('…', r'\ldots{}'),
       ('√', r'$\surd$'), ('σ', r'$\sigma$'), ('ñ', r'\~n')]

SPECIAL = [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
           ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
           ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]


def esc(t):
    for a, b in SPECIAL:
        t = t.replace(a, b)
    for a, b in SYM:
        t = t.replace(a, b)
    return t


def has_image(p):
    return bool(p._element.findall('.//' + W + 'drawing'))


def eq_number(p):
    if p._element.find(M + 'oMath') is None:
        return None
    for r in p._element.findall(qn('w:r')):
        t = r.find(qn('w:t'))
        if t is not None and t.text and re.match(r'^\(\d+\)$', t.text):
            return int(t.text.strip('()'))
    return None


def table_latex(rows, nhead=1):
    ncol = max(len(r) for r in rows)
    lines = [r'\begin{tabular}{l' + ' c' * (ncol - 1) + '}', r'\hline']
    for i, r in enumerate(rows):
        r = r + [''] * (ncol - len(r))
        lines.append(' & '.join(esc(x) for x in r) + r' \\')
        if i == nhead - 1:
            lines.append(r'\hline')
    lines += [r'\hline', r'\end{tabular}']
    return lines


doc = Document(SRC)
body = doc.element.body
paras = doc.paragraphs
par_map = {p._element: p for p in paras}
tab_map = {t._element: t for t in doc.tables}
elements = list(body)

keywords = ''
for p in paras:
    if p.text.strip().startswith('Keywords:'):
        keywords = p.text.strip()[len('Keywords:'):].strip()

out = [r'\documentclass{iopjournal}', '', r'\usepackage{amsmath,amssymb}', '',
       r'\begin{document}', '', r'\articletype{Paper}', '']

fig_n = tab_n = eq_n = 0
pending_caption = None
skip = set()

for idx, el in enumerate(elements):
    if el.tag == qn('w:tbl'):
        t = tab_map.get(el)
        if t is None:
            continue
        tab_n += 1
        cap = pending_caption or ('Table %d' % tab_n)
        pending_caption = None
        rows = [[c.text.strip() for c in r.cells] for r in t.rows]
        nhead = 1
        out += [r'\begin{table}[htbp]', r'\caption{%s}' % esc(cap), r'\centering']
        if tab_n in WIDE:
            h1, h2 = WIDE[tab_n]
            rows = [h1, h2] + rows[1:]
            nhead = 2
            out += [r'\small', r'\setlength{\tabcolsep}{4pt}']
        out += table_latex(rows, nhead)
        out += [r'\label{tab%d}' % tab_n, r'\end{table}', '']
        continue
    if el.tag != qn('w:p'):
        continue
    p = par_map.get(el)
    if p is None:
        continue
    t = p.text.strip()

    if has_image(p):
        fig_n += 1
        cap = ''
        for nxt in elements[idx + 1:]:
            if nxt.tag != qn('w:p'):
                break
            q = par_map.get(nxt)
            if q is not None and q.text.strip():
                cap = q.text.strip()
                skip.add(q._element)
                break
        out += [r'\begin{figure}[htbp]', r'\centering',
                r'\includegraphics[width=0.85\textwidth]{figures/%s}' % FIGS[fig_n - 1],
                r'\caption{%s}' % esc(re.sub(r'^Figure\s*\d+\.\s*', '', cap)),
                r'\label{fig%d}' % fig_n, r'\end{figure}', '']
        continue

    if el in skip or not t:
        continue

    n = eq_number(p)
    if n is not None:
        eq_n += 1
        out += [r'\begin{equation}', EQ[n], r'\end{equation}', '']
        continue

    if idx == 0:
        out += [r'\title{%s}' % esc(t), '',
                r'\author{Author Name$^1$, Author Name$^2$ and Author Name$^{1,*}$}', '',
                r'\affil{$^1$Department, Institution, City, Country}',
                r'\affil{$^2$Department, Institution, City, Country}',
                r'\affil{$^*$Author to whom any correspondence should be addressed.}', '',
                r'\email{name@institution.org}', '']
        if keywords:
            out += [r'\keywords{%s}' % esc(keywords), '']
        continue

    if t == 'ABSTRACT':
        out += [r'\begin{abstract}']
        continue
    if t.startswith('Keywords:'):
        out += [r'\end{abstract}', '']
        continue
    if t == 'References':
        continue
    if re.match(r'^\[\d+\]\s', t):
        continue

    m = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.*)$', t)
    if m:
        depth = m.group(1).count('.')
        cmd = (r'\section', r'\subsection', r'\subsubsection')[min(depth, 2)]
        out += ['%s{%s}' % (cmd, esc(m.group(2))), '']
        continue

    if p.paragraph_format.alignment == WD_ALIGN_PARAGRAPH.CENTER and t.startswith('Table '):
        pending_caption = t
        continue

    out += [esc(t), '']

# close the abstract if the Keywords line was missing
if r'\begin{abstract}' in out and r'\end{abstract}' not in out:
    out += [r'\end{abstract}', '']

refs = []
for p in paras:
    m = re.match(r'^\[(\d+)\]\s*(.+)$', p.text.strip())
    if m:
        refs.append((int(m.group(1)), m.group(2)))
refs.sort()
out += [r'\begin{thebibliography}{99}', '']
for n, txt in refs:
    txt = re.sub(r'\s*\[[A-Z]{1,3}\]\s*\.', '.', txt)   # drop the GB/T type marker
    out += [r'\bibitem{%d} %s' % (n, esc(txt))]
out += ['', r'\end{thebibliography}', '', r'\end{document}']

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('equations %d/%d  figures %d  tables %d  refs %d'
      % (eq_n, len(EQ), fig_n, tab_n, len(refs)))
