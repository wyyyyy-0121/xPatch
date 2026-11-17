# Probability-screen addendum

This addendum accompanies the six-page paper. It reports a frozen, two-fold historical development screen of conditional scale, separate from the original conditional-mean forecasting goal. The 41-page S1–S38 supplement retains its previous bytes and chronology.

## Recipe and information scope

The targets are the original minute H60 and hourly H8 cumulative log returns. Each density has zero location and five Student-t degrees of freedom. B0 fits a constant variance. B1 uses an EWMA with decay 0.94 at native-bar frequency, multiplies by H and estimates one positive fit-prefix calibration factor. This adapts the daily RiskMetrics idea rather than reproducing its original normal-return setup. B2 fits a penalized log-variance regression on short, one-complete-session and twenty-complete-session log squared-return statistics. C adds sine/cosine daily clock coordinates and the calendar-known number of session boundaries in the future path. These calendar features require no future price realization. The two ablations remove clock or gap terms. Minute A_gap is identical to C because its permitted paths do not cross sessions.

All nonlinear-scale fits use mean Student-t negative log likelihood and fixed L2 penalty 0.01 on standardized non-intercept coefficients. Student-t scale squared equals three-fifths of the predicted variance. Fit-only numerical bounds, optimizer settings and temporal identities were frozen in the original execution protocol before scoring. The original fit/stop/evaluation roles and label-maturity purge remain; stop observations are not used for tuning or calibration in this screen.

The common historical-support mask excludes the earliest 6,511 minute fit origins and 31 hourly fit origins in each of the first two folds. It removes no evaluation origin: the minute task retains 151,440 evaluated origins and hourly retains 2,718. These counts overlap in outcome time and do not represent independent market episodes.

## Complete pooled scores

Scores pool origins within each task. NLL uses nat/origin; lower NLL and CRPS are better. Negative continuous-density NLL is valid and depends on the fixed return units. Coverage describes central nominal intervals and does not imply independence.

### Minute H60

|Recipe|NLL|CRPS|50% coverage|90% coverage|50% width|90% width|
|---|---:|---:|---:|---:|---:|---:|
|B0|-3.83249971|0.00286340|46.84%|84.56%|0.00509647|0.01413214|
|B1|-3.92586444|0.00281526|49.29%|90.27%|0.00665317|0.01844874|
|B2|-3.94882612|0.00276854|46.66%|89.33%|0.00560924|0.01555400|
|C|-3.95099486|0.00276693|46.56%|89.35%|0.00560746|0.01554908|
|A clock|-3.94882612|0.00276854|46.66%|89.33%|0.00560924|0.01555400|
|A gap|-3.95099486|0.00276693|46.56%|89.35%|0.00560746|0.01554908|

### Hourly H8

|Recipe|NLL|CRPS|50% coverage|90% coverage|50% width|90% width|
|---|---:|---:|---:|---:|---:|---:|
|B0|-2.36050810|0.01211101|42.46%|82.60%|0.02124024|0.05889761|
|B1|-2.43000226|0.01188305|44.63%|88.34%|0.02456946|0.06812927|
|B2|-2.41355379|0.01194805|40.91%|84.69%|0.02135114|0.05920511|
|C|-2.42284903|0.01191556|40.91%|84.77%|0.02145128|0.05948280|
|A clock|-2.42290989|0.01191521|40.91%|84.92%|0.02144908|0.05947669|
|A gap|-2.42195691|0.01191952|40.95%|84.69%|0.02142965|0.05942282|

## Frozen decision and limits

Each task requires C to beat B0 and B1 in each fold, reduce their pooled NLL by at least 0.01 nat/origin, be no worse than B2 in pooled NLL, and attain 90% coverage within [87%,93%]. Both tasks must pass before the remaining folds can start. Minute passes these screening rules but its 50% coverage is only 46.56%; the small 0.00217 gain over B2 does not meet the separate 0.01 full-development threshold. Hourly fails: C trails B1 in fold 1 and in pooled NLL, and its 90% coverage is 84.77%. The conjunction fails, so the batch stops. No remaining fold, five-fold block interval or unseen confirmation is performed.

The 24 completed recipe positions correspond to 22 unique fits and two minute identities reused without refitting. Independent scalar reconstruction of saved parameters and density evaluations passes, as does a real child-process termination and checkpoint restart under the original deadline. These checks establish the recorded numerical and lifecycle scope, not independent market validation. All results concern previously exposed development periods; Q3 remains unread. The implementation used one process, two logical CPUs and no GPU or paid resource.

Clock/gap ablations supply descriptive comparisons only. The gain of C over the no-clock arm is 0.00216874 at minute H60 and -0.00006086 at hourly H8; its hourly gain over the no-gap arm is 0.00089212 nat/origin. No block intervals support attribution or economic causality. B1 and B2 both improve pooled NLL over B0 for each task, but hourly B1 does not beat B0 in every fold. These data support limited conditional-scale development signals without a successful new method.

## Sources and reproduction

The methodology review obtained and inspected the RiskMetrics technical document, fourth edition (J.P. Morgan/Reuters, 1996), and Bollerslev's original 1987 conditional Student-t paper (doi:10.2307/1925546). Corsi (2009, doi:10.1093/jjfinec/nbp001) supports the multi-window motivation through the checked publisher/author abstract and bibliographic record; publisher full text returned HTTP 403. B2 is HAR-type and is not a replication of standard daily HAR-RV.

The bundled probability_RESULTS.csv contains every fold/recipe score; probability_RESULTS_BY_GAP.csv contains the prespecified calendar grouping. probability_DECISION.json records all gates and resource/timing statements. The replay and recovery receipts define their exact checks. Raw bars, per-origin scientific arrays and checkpoints remain external dependencies of the original experiment. This addendum provides result documentation, not a portable raw-data retraining capsule.
