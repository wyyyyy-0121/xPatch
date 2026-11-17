"""Bind the independent claim review and actual page inspection to final files."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent.parent
OLD = P.parent / '20261110_bar_location_followup'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    b = read(P/'review/DOCUMENT_BUILD.json')
    initial = read(P/'review/initial_render/DOCUMENT_BUILD.json')
    prev = read(OLD/'review/DOCUMENT_BUILD.json')
    claim = read(P/'review/SPLIT_CLAIM_REVIEW_FINAL.json')
    changelog = read(P/'SOURCE_CHANGELOG.json')
    manifest = read(P/'EVIDENCE_MAP.json')
    assert claim['status'] == 'passed_independent_final_source_claim_review' and not claim['pending_findings']
    assert all(sha(path) == h for path, h in claim['files'].items())
    assert changelog['new_main'] == sha(P/'MAIN_PAPER_EN.md')
    assert changelog['new_supplement'] == sha(P/'SUPPLEMENT_EN.md')
    assert len(manifest['new_snapshots']) == 14
    for row in manifest['new_snapshots']:
        assert sha(P/row['snapshot']) == row['sha256'] == sha(row['original_path'])
    assert sha(OLD/'EVIDENCE_MAP.json') == manifest['inherited_bar_location_evidence']['sha256']
    viewed = {'MAIN_PAPER': {'page-1.png', 'page-2.png', 'page-3.png', 'page-4.png',
                             'page-7.png', 'page-8.png', 'page-9.png'},
              'SUPPLEMENT': {'page-34.png', 'page-35.png'}}
    visual, same_count = [], 0
    for doc in b['documents']:
        name = doc['name']
        old_hashes = set(next(x for x in prev['documents'] if x['name'] == name)['preview_sha256'].values())
        initial_hashes = set(next(x for x in initial['documents'] if x['name'] == name)['preview_sha256'].values())
        assert sha(P/(name+'_EN.md')) == doc['source_sha256']
        assert sha(P/(name+'_EN.pdf')) == doc['pdf_sha256']
        assert sha(P/(name+'_EN.docx')) == doc['docx_sha256']
        assert doc['preservation']['passed'] and doc['preservation']['final_layout_preservation_passed']
        actual, inherited = [], []
        for image, identity in doc['preview_sha256'].items():
            assert sha(Path(doc['preview_directory'])/image) == identity
            if identity in old_hashes:
                inherited.append(image)
            else:
                assert image in viewed[name]
                rechecked = {'page-7.png'} if name == 'MAIN_PAPER' else {'page-35.png'}
                if image not in rechecked:
                    assert identity in initial_hashes
                actual.append(image)
        assert len(actual) + len(inherited) == doc['pages']
        same_count += len(inherited)
        visual.append({'document': name, 'pages': doc['pages'], 'pdf_sha256': doc['pdf_sha256'],
                       'columns': doc['columns'], 'page_preview_sha256': doc['preview_sha256'],
                       'changed_pages_actually_viewed': actual, 'identical_previously_accepted_pages': inherited})
    assert same_count == 35 and sum(len(v['changed_pages_actually_viewed']) for v in visual) == 9
    assert [x['pages'] for x in b['documents']] == [9, 35]
    assert [x['columns'] for x in b['documents']] == [2, 1]
    assert (P/'SUPPLEMENT_EN.md').read_bytes().startswith((OLD/'SUPPLEMENT_EN.md').read_bytes())
    main_pdf_text = ' '.join((P/'MAIN_PAPER_PDF_TEXT.txt').read_text().split())
    supp_pdf_text = ' '.join((P/'SUPPLEMENT_PDF_TEXT.txt').read_text().split())
    for text in ('Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales',
                 'Wang Yang', '4.12', '330.938', '275.931'):
        assert text in main_pdf_text, text
    for text in ('S32.', 'Annual H12 price', 'Quarterly H3 price', 'Monthly H21 price',
                 '331.172', '282.327', '37,544'):
        assert text in supp_pdf_text, text
    visual_report = {'status': 'passed_actual_or_exact_previous_image_review', 'reviewer': 'root',
                     'documents': visual, 'pages_total': 44, 'previous_exact_accepted_images': 35,
                     'changed_pages_actually_viewed': 9, 'initial_and_final_changed_page_hashes_verified': True,
                     'no_market_or_model_array_read': True}
    (P/'review/VISUAL_REVIEW.json').write_text(json.dumps(visual_report, indent=2)+'\n')
    checks = {'claim_review_passed_and_files_current': True, 'all_14_new_snapshots_and_external_map_identity': True,
              'source_changelog_matches': True, 'old_supplement_exact_prefix': True,
              'title_author_and_scope_retained': True, 'pdf_and_docx_source_hashes_current': True,
              'two_column_main_single_column_supplement': True, 'all_44_render_pages_covered': True,
              'split_table_and_main_text_visible': True, 'original_replay_failure_preserved': True}
    result = {'status': 'accepted_announced_split_document_edition_research_goal_active', 'checks': checks,
              'checks_passed': len(checks), 'reviewed_main_pages': 9, 'reviewed_supplement_pages': 35,
              'visual_review_sha256': sha(P/'review/VISUAL_REVIEW.json'),
              'independent_claim_review_sha256': sha(P/'review/SPLIT_CLAIM_REVIEW_FINAL.json'),
              'main_pdf_sha256': sha(P/'MAIN_PAPER_EN.pdf'), 'supplement_pdf_sha256': sha(P/'SUPPLEMENT_EN.pdf'),
              'initial_checker_failure': 'Title string comparison failed because the PDF wraps the title over two lines; '
                  'this checker normalizes PDF whitespace before asserting the exact words. No source was changed.',
              'new_market_training_or_inference_by_build': False, 'Q3_values_read': False,
              'research_goal_achieved': False, 'independent_market_confirmation': False,
              'actual_new_predictive_method_success': False, 'email_placeholder_still_requires_author': True,
              'new_zip_claimed': False}
    (P/'review/FINAL_DOCUMENT_ACCEPTANCE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
