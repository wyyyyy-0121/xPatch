"""Prepare a six-page documentary edition from reviewed manuscript/results.

No market readers, model inference, fitting, or bootstrap execution.
"""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import re
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
PARENT = BASE.parent / '20261111_contribution_revision'
SCREEN = ROOT / 'outputs/multiscale/probability_direction_screen/20261111_fast_screen'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def latex(text):
    return subprocess.run(['pandoc', '-f', 'markdown+tex_math_single_backslash+autolink_bare_uris',
                           '-t', 'latex', '--wrap=none'], input=text,
                          text=True, check=True, capture_output=True).stdout

def cite(m):
    raw = m.group(1).replace('–', '-').replace('—', '-')
    nums = []
    for part in raw.split(','):
        if '-' in part:
            a,b = map(int, part.split('-')); nums.extend(range(a,b+1))
        else: nums.append(int(part))
    return r'\cite{' + ','.join('r'+str(x) for x in nums) + '}'

def main():
    original = (PARENT / 'MAIN_PAPER_EN.md').read_text()
    text, references = original.split('## References', 1)
    # Keep the front matter in main.tex. Abstract becomes a regular unnumbered
    # heading so its text occupies the same two-column stream as the paper.
    text = text.split('## Abstract\n', 1)[1]
    abstract, text = text.split('## 1. Introduction and related work', 1)
    text = '## 1. Introduction and related work' + text
    text = text.replace('## 2. Data, tasks and evaluation\n',
        '## 2. Data, tasks and evaluation\n\n### 2.1 Targets, clocks and source information\n')
    text = text.replace('Five expanding folds purge fit and stopping labels',
        '### 2.2 Temporal separation and loss aggregation\n\nFive expanding folds purge fit and stopping labels')
    insertion = r'''

Let \(\tau_t\) be the decision time and \(\ell(t,H)\) the completion time of the last target bar. A fit origin can precede the next partition in index order while its outcome still extends into that partition. We therefore exclude origins with \(\ell(t,H)\geq\tau_{\mathrm{next}}\) at each applicable boundary. This purge concerns label availability; it does not erase the exposure of an assessment period during later research. Figure @figdesign summarizes the information flow.

The distinction between trading bars and clock time matters most at hourly H8. The retained series contains only complete hourly bars. A path that crosses a session includes the intervening overnight change and any omitted hourly remainder in the next retained close-to-close increment. Minute H60 paths remain within one regular session. These targets consequently differ in both elapsed time and information arrival, even though both use cumulative log returns.

For a predictor with seeds \(s=1,\ldots,S\), we compute \(L_t=S^{-1}\sum_s(y_t-\hat y_{t,s})^2\), then pool the retained origins. Squaring an averaged prediction would instead evaluate an ensemble. We use the former rule throughout the reported neural comparisons. Repeating seeds measures sensitivity to training randomness; overlapping origins and repeated seeds do not increase the number of independent market episodes. The bootstrap keeps the pairing between a method and its reference and uses the stated chronological block groups.

FIGDESIGN

'''
    text = text.replace('## 3. Controlled implementation study', insertion+'## 3. Controlled implementation study')
    text = text.replace('Predictive success remains stricter than repairing a weak default:',
        r'''Initialization and calibration affect different stages of the procedure. Initializing the output head changes the starting function before any optimization, while fitting \(\alpha\) changes the amplitude of an already trained forecast using the inner stopping tail. The four-cell design keeps these roles distinct. A zeroed head also suppresses the first backbone gradient, so the intervention includes an early optimization effect; it cannot identify the starting function as the sole mediator. The continued default checkpoints preserve their optimizer and random state, making the training-cap comparison a continuation rather than a second independent fit.

Predictive success remains stricter than repairing a weak default:''')
    # Remove the two Markdown tables and their captions; reinsert native floats.
    p1 = text.index('**Table 1.**'); p2 = text.index('![Figure 1.')
    text = text[:p1] + 'TABLETASKS\n\nTABLECAP\n\nFIGCAP\n\n' + text[p2:]
    text = re.sub(r'!\[Figure 1\..*?\]\(figures/return_controls.png\)\{width=3.3in\}',
                  'FIGRETURNS', text, flags=re.S)
    text = text.replace('Table 2 shows', 'Table @tabcap shows').replace('Table 1 reports','Table @tabtasks reports')
    text = text.replace('Both contrasts improve in all five fold means',
        'Figure @figreturns compares these recipes. Both contrasts improve in all five fold means')
    text = text.replace('Persistence MSE is 1.8985 dollars squared.',
        'Figure @figcap plots this comparison. Persistence MSE is 1.8985 dollars squared.')
    text = text.replace('Quarterly-H3 returns provide',
        'Figure @figannual compares the annual repair with the shrunk mean on all origins and on the original nonoverlapping subset. The four-path subset is a sensitivity check, not an independent test or an estimate of a population ranking.\n\nFIGANNUAL\n\nQuarterly-H3 returns provide')
    prob = r'''

### 4.5 Limited probability-forecasting screen

We also completed a separately frozen two-fold development screen for the original minute H60 and hourly H8 targets. The return location remained zero. B0 fits a constant scale; B1 uses an EWMA with fixed decay 0.94 and a fit-prefix scale calibration [20]; B2 uses short, daily and twenty-session volatility summaries in a HAR-type log-variance regression [21]. Candidate C adds daily sine/cosine clock coordinates and the calendar-known count of session boundaries. This is a multi-window motivation, not a reproduction of the standard daily HAR-RV model. All arms use a Student-t density with five degrees of freedom. We map its scale \(s\) to variance \(v\) by \(s^2=3v/5\).

The fitting rule, temporal purge, common support mask and evaluation origins remain fixed before scoring. The added history requirement removes the earliest unsupported fit origins but no evaluation origins. The six registered recipes include two ablations; two minute positions reuse C because the gap count is zero. We complete 24 positions with 22 unique fits. This stage uses NLL and CRPS to assess a predictive distribution rather than a conditional-mean MSE. Continuous-density NLL can be negative; comparisons below use paired differences within one task and the same return units.

TABLEPROB

C passes the minute screen: it reduces pooled NLL by 0.02513 nat/origin against B1 and by 0.00217 against B2, with 89.35% coverage for a nominal 90% interval. Its 50% coverage is 46.56%, so this pass does not establish full calibration. At hourly H8, C trails B1 by 0.00715 nat/origin and covers only 84.77% with its 90% interval. The frozen rule requires both tasks to pass; we therefore stop without the remaining three folds, block intervals or independent confirmation. B1 and B2 both beat B0 in pooled NLL on each task, but hourly B1 does not win in every fold. These are descriptive conditional-scale signals, with no established new-method or mean-forecast advantage. The probability-screen addendum records the full scores and decision.

'''
    text = text.replace('## 5. Discussion and conclusion',prob+'## 5. Discussion and conclusion')
    text = text.replace('The twenty-task result distinguishes this repair',
        '### 5.1 Interpretation of the supported effects\n\nThe twenty-task result distinguishes this repair')
    text = text.replace('All assessment periods had been inspected during development.',
        '### 5.2 Validity, transfer and research implications\n\nAll assessment periods had been inspected during development.')
    text = text.replace('For this AAPL study, we recommend reporting',
        '''The probability screen illustrates a separate research question. A scale forecast can improve density scores while the best location forecast remains zero. Such an improvement concerns uncertainty assessment and would not establish directional predictability, trading profits or an advantage in the original return-MSE task. The candidate's failure at H8 also prevents extrapolating the minute result across frequencies. Selecting the minute arm after observing this outcome would create a new development choice; any further study would require its own frozen scope and confirmation protocol.

Transfer to other equities would require matching corporate-action handling, market hours, sampling conventions and target construction. A second stock or a new calendar period would test a broader question than replaying the present checkpoints. Similarly, more annual seeds cannot repair the shortage of nonoverlapping annual paths. The present evidence supports reporting these limitations alongside the measured implementation effects rather than treating successful numerical reconstruction as independent financial validation.

For this AAPL study, we recommend reporting''')
    text = text.replace('## Data and code availability', '## Data and code availability')
    text = text.replace('maps this focused main paper to the unchanged 41-page supplement and preceding accepted computation records.',
        'maps this six-page edition to the unchanged 41-page supplement, the probability-screen addendum and preceding accepted computation records.')
    text = text.replace('No external submission, public deposit or archival DOI has been created.',
        'The LaTeX package contains editable sources, vector figures and aggregate plotting values; full retraining retains the recorded external data and checkpoint dependencies. No external submission, public deposit or archival DOI has been created.')
    text = text.replace('Our contribution concerns the evidence obtained when these choices enter the same comparison. First, we measure initialization and anchor interventions with matched architectures, data, seeds and training rules, and compare initialization with extending the default training cap. The cross-seed return-error reductions are relative to the default implementation; we also report performance against zero. Second, we apply one fixed initialization-and-shrinkage recipe to all twenty tasks and report stronger mean and Ridge references, outcome overlap and event concentration alongside the gains. This separates implementation repair from an incremental forecasting advantage. Third, we document two specific interpretation limits: a public split announcement changes the monthly reference substantially, and a one-coordinate LayerNorm removes input dependence from a short-horizon trend substream. We state the affected tasks for each finding.',
        'We contribute three controlled comparisons. First, we measure initialization and anchor effects under matched architectures, data, seeds and training rules, and contrast initialization with extending the default training cap. Second, we apply one fixed recipe to all twenty tasks, alongside mean and Ridge references, outcome overlap and event concentration. Third, we document how a public split announcement changes the monthly reference and how one-coordinate LayerNorm restricts a short-horizon trend substream. Each finding retains its task-specific scope.')
    text = text.replace('We retain the eight-method comparison of persistence, drift, seasonal repetition, Ridge, MLP, GRU, LSTM and adapted xPatch. The additional evidence comes from the controlled implementation contrasts and their cross-scale evaluation; zero-function initialization and scalar shrinkage are familiar constructions. The assessment periods have contributed to development decisions, so our results describe exposed historical evaluations. They establish neither a new forecasting architecture nor independent market confirmation.',
        'The eight methods are persistence, drift, seasonal repetition, Ridge, MLP, GRU, LSTM and adapted xPatch. Zero-function initialization and shrinkage are established constructions. We assess their implementation effects on development-exposed history, without claiming a new architecture or independent confirmation.')
    text = text.replace('Transfer to other equities would require matching corporate-action handling, market hours, sampling conventions and target construction. A second stock or a new calendar period would test a broader question than replaying the present checkpoints. Similarly, more annual seeds cannot repair the shortage of nonoverlapping annual paths. The present evidence supports reporting these limitations alongside the measured implementation effects rather than treating successful numerical reconstruction as independent financial validation.',
        'Transfer to other equities requires matching corporate actions, market hours and targets. New stocks or periods would test broader generalization; replayed checkpoints and additional seeds cannot provide new market episodes.')
    text = text.replace('The probability screen illustrates a separate research question. A scale forecast can improve density scores while the best location forecast remains zero. Such an improvement concerns uncertainty assessment and would not establish directional predictability, trading profits or an advantage in the original return-MSE task. The candidate\'s failure at H8 also prevents extrapolating the minute result across frequencies. Selecting the minute arm after observing this outcome would create a new development choice; any further study would require its own frozen scope and confirmation protocol.',
        'Scale forecasts can improve density scores while the location forecast remains zero. This concerns uncertainty, not directional or trading gains. The H8 failure prevents a cross-frequency claim. Continuing only the minute arm after this outcome would require a new frozen development scope and confirmation protocol.')
    start = text.index('The [reproduction guide]')
    text = text[:start] + '''The reproduction guide links this edition to the unchanged 41-page supplement, probability addendum and original computation records. The package includes editable LaTeX, vector figures and plotting aggregates. Full numerical reproduction requires the recorded external data and checkpoints. Computational replay supplies no independent market evidence. The author will complete the email before submission. No external submission, public deposit or archival DOI has been created.
'''
    text = text.replace('The eight methods are persistence, drift, seasonal repetition, Ridge, MLP, GRU, LSTM and adapted xPatch.',
        'The eight methods are persistence, drift, seasonal repetition, Ridge, MLP, GRU, LSTM and adapted xPatch. Prior sequence-model comparisons [13] provide a broader context for these neural benchmarks.')
    refs = re.findall(r'^\[(\d+)\]\s+(.*)$', references, flags=re.M)
    refs += [('20','J.P. Morgan/Reuters. *RiskMetrics—Technical Document*, 4th ed., 1996. https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a'),
             ('21','Corsi, F. “A Simple Approximate Long-Memory Model of Realized Volatility.” *Journal of Financial Econometrics*, 7(2), 174–196, 2009. https://doi.org/10.1093/jjfinec/nbp001')]
    # No local .md link in the finished PDF; provenance belongs in the package.
    text = text.replace('[reproduction guide](REPRODUCTION_GUIDE_EN.md)', 'reproduction guide')
    (BASE/'MANUSCRIPT_EN.md').write_text('## Abstract\n\n'+abstract+text+'\n## References\n\n'+ '\n\n'.join('['+n+'] '+t for n,t in refs)+'\n')
    text = re.sub(r'^## (?:\d+\. )?(.+)$',r'# \1',text,flags=re.M)
    text = re.sub(r'^### \d+\.\d+ (.+)$',r'## \1',text,flags=re.M)
    # Abstract without Pandoc's standalone abstract treatment.
    # Citations follow a prose space; math intervals such as _{[0,1]}
    # must not become references. Convert before Pandoc escapes brackets.
    cited = re.sub(r'(?<= )\[(\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*)\]',
                   cite, abstract+'\n'+text)
    converted = latex(cited)
    converted = converted.replace('Keywords:', 'Keywords:')
    tokens = {
      'FIGDESIGN':r'''\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{figures/study_design.pdf}
\caption{Study design. Each intervention retains the matched task and temporal identity. Fixed cross-scale scoring includes strong references; numerical replay does not provide an unseen market period.}\label{fig:design}
\end{figure}''',
      'TABLETASKS':r'''\begin{table}[t]
\centering\small
\caption{Fixed initialization-and-shrinkage recipe: endpoint-MSE improvement over persistence/zero. P/R denote seed counts. The six long-scale entries use the post-observation replication.}\label{tab:tasks}
\begin{tabular}{lrrc}\toprule
Task & Price & Return & Seeds P/R\\\midrule
Annual H12 & +33.731\% & +38.105\% & 3/3\\
Quarterly H3 & +1.226\% & +0.698\% & 3/3\\
Monthly H21 & +0.063\% & +0.741\% & 3/3\\
Hourly H1 & $-0.209$\% & $-0.796$\% & 3/1\\
Hourly H4 & $-0.061$\% & $-0.517$\% & 1/1\\
Hourly H8 & $-1.636$\% & $-0.795$\% & 1/3\\
Minute H5 & $-0.011$\% & $-0.010$\% & 1/1\\
Minute H15 & $-0.035$\% & +0.003\% & 1/1\\
Minute H30 & $-0.138$\% & $-0.339$\% & 1/1\\
Minute H60 & $-0.066$\% & $-0.038$\% & 1/3\\\bottomrule
\end{tabular}\end{table}''',
      'TABLECAP':r'''\begin{table}[t]\centering\small
\caption{Hourly-H1 mean-seed price MSE (USD$^2$). Default-60 continues Default-20; Initialized-20 retains the direct-price target. Persistence MSE is 1.8985.}\label{tab:cap}
\begin{tabular}{lrrr}\toprule
Model & Default-20 & Default-60 & Init.-20\\\midrule
LSTM & 769.51 & 269.68 & 1.9039\\
xPatch & 6494.47 & 4641.93 & 1.9053\\\bottomrule
\end{tabular}\end{table}''',
      'FIGCAP':r'''\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{figures/cap_comparison.pdf}
\caption{Training-cap and initialization comparison from Table~\ref{tab:cap}, on a logarithmic MSE axis. The dashed persistence line distinguishes repair of a weak default from beating the price reference.}\label{fig:cap}
\end{figure}''',
      'FIGRETURNS':r'''\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{figures/return_controls.pdf}
\caption{Matched xPatch return recipes. Zero-return MSE equals one; each bar pools per-origin mean seed loss. Only ratios below one beat zero. The factorial cells and calibrated initialized recipe remain distinct.}\label{fig:returns}
\end{figure}''',
      'FIGANNUAL':r'''\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{figures/annual_controls.pdf}
\caption{Annual repair and shrunk mean: all 47 overlapping origins versus the original four nonoverlapping paths. These are dependent summaries of the same history, without confidence intervals or independent confirmation.}\label{fig:annual}
\end{figure}''',
      'TABLEPROB':r'''\begin{table}[t]\centering\small
\caption{Two-fold probability screen. Positive $\Delta$NLL means that C has lower NLL than the reference (nat/origin). Coverage reports C's central intervals. No five-fold or independent confirmation follows.}\label{tab:probability}
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrr}\toprule
Quantity & Minute H60 & Hourly H8\\\midrule
Evaluation origins & 151,440 & 2,718\\
$\Delta$NLL vs B0 & +0.11850 & +0.06234\\
$\Delta$NLL vs B1 & +0.02513 & $-0.00715$\\
$\Delta$NLL vs B2 & +0.00217 & +0.00930\\
50\% coverage & 46.56\% & 40.91\%\\
90\% coverage & 89.35\% & 84.77\%\\
Task screen & Pass & Fail\\\bottomrule
\end{tabular}\end{table}'''
    }
    for key,value in tokens.items():
        assert converted.count(key)==1,(key,converted.count(key))
        converted=converted.replace(key,value)
    for marker,label in [('figdesign','fig:design'),('tabcap','tab:cap'),('tabtasks','tab:tasks'),('figannual','fig:annual'),('figcap','fig:cap'),('figreturns','fig:returns')]:
        converted=converted.replace('@'+marker,r'\ref{'+label+'}')
    # Ensure single-column equations do not exceed the measure.
    converted=converted.replace(r'\hat y=b+\alpha(f-b),\quad',r'\begin{gathered}\hat y=b+\alpha(f-b),\\')
    converted=converted.replace(r'\sum q^2}\right).',r'\sum q^2}\right).\end{gathered}')
    converted=re.sub(r'(?<![\w])-(\d+\.\d+)\\%',r'$-\1$\\%',converted)
    (BASE/'sections/paper.tex').write_text(r'\section*{Abstract}'+'\n'+converted)
    bib=['\\begin{thebibliography}{99}','\\small\\setlength{\\itemsep}{0pt}\\setlength{\\parskip}{0pt}']
    for n,t in refs: bib += [r'\bibitem{r'+n+'} '+latex(t).strip()]
    bib += [r'\end{thebibliography}']
    (BASE/'references.tex').write_text('\n\n'.join(bib)+'\n')
    shutil.copyfile(PARENT/'figures/return_controls.pdf',BASE/'figures/return_controls.pdf')
    for name,new in [('SUPPLEMENT_EN.pdf','S1_S38_previous.pdf'),('SUPPLEMENT_EN.md','S1_S38_previous.md')]:
        shutil.copyfile(PARENT/name,BASE/'supplement'/new)
    for name in ['RESULTS.csv','RESULTS_BY_GAP.csv','DECISION.json','INDEPENDENT_REPLAY.json','RECOVERY_TEST.json']:
        shutil.copyfile(SCREEN/name,BASE/'supplement'/('probability_'+name))
    shutil.copyfile(SCREEN/'FINAL_RESULTS_ZH.md',BASE/'supplement/probability_results_ZH.md')
    locks={str(p.relative_to(ROOT)):sha(p) for p in [PARENT/'MAIN_PAPER_EN.md',PARENT/'MAIN_PAPER_EN.pdf',PARENT/'SUPPLEMENT_EN.pdf',SCREEN/'DECISION.json',SCREEN/'RESULTS.csv',
           ROOT/'outputs/multiscale/mechanism_repair/20261109_v3a/longscale_replication/analysis/POOLED.csv']}
    (BASE/'review/SOURCE_LOCKS.json').write_text(json.dumps(locks,indent=2)+'\n')
    (BASE/'review/SOURCE_EXPANSION.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),(BASE/'MANUSCRIPT_EN.md').read_text().splitlines(True),fromfile='contribution_revision',tofile='six_page_edition')))
    print('Expanded main words:',len((abstract+text).split()),'; references:',len(refs))

if __name__=='__main__':main()
