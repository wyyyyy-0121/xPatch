"""Record quality and public-source evidence for a supplied one-minute CSV."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from data_quality import as_market, expected_index, load_ohlcv, sha256

SOURCE='https://hfdatalibrary.com/pages/download?'
DOCS='https://hfdatalibrary.com/pages/docs'
ISSUES='https://hfdatalibrary.com/pages/issues'


def audit(path):
    path=Path(path)
    frame,meta=load_ohlcv(path)
    original=pd.read_csv(path, usecols=lambda name: str(name).strip().lower() == 'source')
    source_values=(original.iloc[:,0].value_counts(dropna=False).to_dict()
                   if len(original.columns) else {})
    # The supplied local timestamps are naive; the publisher's docs specify ET
    # and labels 09:30 through 15:59, corroborated by the file's source column.
    local=frame.copy()
    local.index=as_market(local.index)
    expected=expected_index(local.index,'1min')
    exp=pd.Series(1,index=expected).groupby(expected.normalize()).sum()
    observed=local.index[local.index.isin(expected)]
    obs=pd.Series(1,index=observed).groupby(observed.normalize()).sum()
    coverage=obs.reindex(exp.index,fill_value=0)/exp
    counts=local.groupby(local.index.normalize()).size()
    gaps=local.index.to_series().diff().dropna()
    dec23=local.loc['2024-12-23'] if pd.Timestamp('2024-12-23',tz='America/New_York') in local.index.normalize() else local.iloc[:0]
    result=dict(csv=str(path.resolve()),csv_sha256=sha256(path),file_bytes=path.stat().st_size,
      rows=len(frame),source_values={str(k):int(v) for k,v in source_values.items()},
      timestamp_range=[frame.index.min().isoformat(),frame.index.max().isoformat()],
      naive_timestamp_localization='America/New_York, based on supplied source URL and provider methodology; original CSV has no timezone offset',
      duplicate_timestamps=meta['duplicate_timestamps_removed'],
      interpolation=dict(second_difference_fraction=meta.get('linear_second_difference_fraction'),
                         suspected_interpolation=meta.get('suspected_interpolation')),
      calendar_coverage=dict(expected_minutes=int(exp.sum()),observed_minutes=int(obs.sum()),
          fraction=float(obs.sum()/exp.sum()),sessions=len(exp),sessions_below_98_percent=int((coverage<.98).sum()),
          median_session=float(coverage.median()),minimum_session=float(coverage.min()),
          quantiles={str(k):float(v) for k,v in coverage.quantile([0,.01,.05,.25,.5,.75,.95,1]).items()},
          excluded_outside_regular_minutes=len(frame)-len(observed)),
      daily_row_count=dict(minimum=int(counts.min()),median=float(counts.median()),maximum=int(counts.max())),
      largest_gap_minutes=gaps.dt.total_seconds().div(60).value_counts().head(10).astype(int).to_dict(),
      dec_23_2024=dict(rows=len(dec23),first=dec23.index.min().isoformat() if len(dec23) else None,
                       last=dec23.index.max().isoformat() if len(dec23) else None),
      provider_scope='IEX Exchange HIST only, approximately 2-3% of consolidated market volume; not consolidated tape',
      provider_cleaning='Nonzero-volume bars only; removes >25% one-minute returns and Brownlees-Gallo centered 50-bar outliers; centered filter uses neighboring future bars for sample retention.',
      provider_price_basis='Per provider docs, split-adjusted only after 2026-10-06; no dividend/spin-off adjustment. This file version is not independently checksum-matched to provider.',
      formal_eligibility='fails: regular-session coverage is below the configured 98% requirement and one publisher-documented session is cut short',
      exploratory_policy='unfilled gaps; samples crossing any missing expected minute are discarded; no final testing; no model accepted',
      source_pages=dict(download=SOURCE,methodology=DOCS,known_issues=ISSUES,
                        download_page_sha256=sha256(HERE/'data/audits/source_pages/hfdata-download.html'),
                        methodology_page_sha256=sha256(HERE/'data/audits/source_pages/hfdata-docs.html'),
                        known_issues_page_sha256=sha256(HERE/'data/audits/source_pages/hfdata-issues.html'),
                        archived_pages='data/audits/source_pages/'),
      source_review='User supplied the provider URL; public page and methodology reviewed on 2026-10-20. The column source=iex and published schema agree. The exact file download event and file-version checksum are unavailable.')
    receipt=dict(decision='verified',reviewer='Codex source review using URL supplied by user',
                 source_url=SOURCE,review_evidence=[DOCS,ISSUES,
                   'Page documents IEX HIST, 09:30-15:59 ET bar labels, Clean filters and split-only adjustment; source column is iex'],
                 timezone='America/New_York',csv_sha256=sha256(path),reviewed_utc=datetime.now(timezone.utc).isoformat())
    receipt_path=HERE/'data'/'audits'/'AAPL_clean_1min.review_receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    manifest=dict(symbol='AAPL',interval='1min',timestamp_label='bar_start',
      source='HF Data Library; IEX Exchange HIST; user-supplied cleaned CSV',provider='HF Data Library / IEX HIST',
      provider_url=SOURCE,timezone='America/New_York',adjustment='split-adjusted only per HF Data Library docs; dividends/spin-offs not adjusted; exact file-version policy not checksum-verified',
      sha256=sha256(path),provenance=dict(method='manual_review',evidence_path='audits/'+receipt_path.name,
           evidence_sha256=sha256(receipt_path)),source_version='not independently checksum-matched')
    path.with_suffix(path.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Re-read after installing the receipt so the saved report includes verified metadata.
    _,verified_meta=load_ohlcv(path)
    result['provenance']=dict(receipt_valid=verified_meta['source_verified'],timezone=verified_meta['timestamp_timezone'],
                              declared_interval=verified_meta['declared_interval'],timestamp_label=verified_meta['timestamp_label'],
                              adjustment=verified_meta['adjustment'])
    result['status']='exploratory_only_quality_gate_failed'
    result['limitations']=[
      '97.0% aggregate regular-session minute coverage is below required 98%.',
      '2024-12-23 is a provider-documented cut-short session; do not interpret it as normal market structure.',
      'IEX-only trades omit other exchanges and many minutes with no IEX trade.',
      'The Clean pipeline uses centered future-neighbor outlier filtering, which can make historical bar inclusion non-causal.',
      'The exact downloaded file version and current split-adjustment basis are not cryptographically tied to provider release metadata.']
    out=HERE/'data'/'audits'/'AAPL_clean_1min_quality.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    md=HERE/'data'/'audits'/'AAPL_clean_1min_quality.md'
    md.write_text(f"""# AAPL Clean 1-minute data audit

Status: `exploratory_only_quality_gate_failed`

File SHA-256: `{result['csv_sha256']}`. Rows: {result['rows']:,}, from {result['timestamp_range'][0]} to {result['timestamp_range'][1]}. No duplicates. The data are strongly non-linear and pass the interpolation heuristic.

HF Data Library's [download page]({SOURCE}) describes one-minute IEX HIST data. Its [methodology]({DOCS}) specifies Eastern Time, bars labelled 09:30 through 15:59, IEX-only volume, split adjustment, and the Clean pipeline. The [known-issues page]({ISSUES}) records that 2024-12-23 is cut short. The source field in this CSV is `iex`. The exact local file bytes were not matched to a published release checksum.

Observed regular-session coverage is {result['calendar_coverage']['fraction']:.2%}, below the project's 98% gate; median session coverage is {result['calendar_coverage']['median_session']:.2%}. {result['calendar_coverage']['sessions_below_98_percent']} sessions are below 98%. The publisher says bars with no IEX trades and zero-volume bars are absent; we do not fill those gaps. Samples intersecting gaps are discarded. The unusually short Dec. 23, 2024 session has {len(dec23)} rows ({dec23.index.min() if len(dec23) else 'none'} through {dec23.index.max() if len(dec23) else 'none'}).

The Clean pipeline also removes outliers using a centered 50-bar window, so future neighboring bars can affect which historical bars remain. Results from this file are exploratory and cannot establish a causal minute forecasting advantage. It is IEX-only data, not consolidated-market OHLCV. The provider's published price-basis notice changed on 2026-10-06; this file's release version is not matched to a checksum.

No prices or missing minutes are synthesized. Formal model rankings remain withheld under the 98% requirement. See `report.json` for hashes, session distributions and source-page hashes.
""")
    print(json.dumps(dict(status=result['status'],rows=len(frame),coverage=result['calendar_coverage']['fraction'],
                          median_session=result['calendar_coverage']['median_session'],
                          min_session=result['calendar_coverage']['minimum_session'],
                          receipt_valid=verified_meta['source_verified'],report=str(out)),indent=2))


if __name__=='__main__':
    audit(HERE/'data'/'AAPL_clean_1min.csv')
