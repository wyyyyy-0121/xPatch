"""Read-only comparison of partition metadata, current arrays and v3 release."""
from pathlib import Path
import hashlib
import json
import os
import time
import zipfile
from io import BytesIO
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PRIOR = REPO/'outputs/multiscale/controlled_followup/20261108_v2'
V3 = REPO/'outputs/multiscale/mechanism_repair/20261109_v3'
RELEASE = V3/'release/AAPL_MULTISCALE_MECHANISM_REVIEW'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indices_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def main():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    started = time.time()
    manifest = read(RELEASE/'MANIFEST.json')['files']
    protocol = read(PRIOR/'protocol/DATA.json')
    rows = []
    with zipfile.ZipFile(V3/'release/AAPL_MULTISCALE_MECHANISM_REVIEW.zip') as package:
        names = set(package.namelist())
        for task in ('minute_h60_future_log_return', 'hourly_h8_future_log_return'):
            taskpath = PRIOR/'cache'/task/'TASK.json'
            meta = read(taskpath)
            release_task = RELEASE/'partitions'/task/'TASK.json'
            assert sha(taskpath) == protocol['task_files'][task]
            assert sha(taskpath) == sha(release_task)
            assert sha(release_task) == manifest[str(release_task.relative_to(RELEASE))]
            for fold in range(1, 6):
                rec = meta['folds'][str(fold)]
                current = Path(rec['split_file'])
                released = RELEASE/'partitions'/task/f'fold_{fold}.npz'
                relative = str(released.relative_to(RELEASE))
                member = 'AAPL_MULTISCALE_MECHANISM_REVIEW/'+relative
                if member not in names:
                    member = relative
                packaged = package.read(member)
                row = {'task': task, 'fold': fold,
                       'current_file': str(current), 'release_file': str(released),
                       'split_metadata_type': type(rec['split_sha256']).__name__,
                       'current_npz_sha256': sha(current), 'release_npz_sha256': sha(released),
                       'release_manifest_sha256': manifest[relative],
                       'packaged_npz_sha256': hashlib.sha256(packaged).hexdigest(),
                       'arrays': {}}
                with np.load(current, allow_pickle=False) as a, np.load(released, allow_pickle=False) as b, np.load(BytesIO(packaged), allow_pickle=False) as c:
                    assert set(a.files) == set(b.files) == set(c.files) == set(rec['split_sha256'])
                    row['keys_equal'] = True
                    for key in sorted(a.files):
                        x, y, z = a[key], b[key], c[key]
                        equal = x.dtype == y.dtype == z.dtype and x.shape == y.shape == z.shape and np.array_equal(x, y) and np.array_equal(x, z)
                        index_hash = indices_sha(x)
                        expected = rec['split_sha256'][key]
                        counts_equal = len(x) == rec['counts'][key]
                        assert equal and index_hash == expected and counts_equal
                        row['arrays'][key] = {'dtype': x.dtype.str, 'shape': list(x.shape),
                            'array_sha256': index_hash, 'metadata_sha256': expected,
                            'values_and_dtype_equal_to_release_and_zip': equal,
                            'count_equals_metadata': counts_equal}
                row['npz_bytes_equal_to_release_and_zip'] = len(set(row[k] for k in ('current_npz_sha256', 'release_npz_sha256', 'release_manifest_sha256', 'packaged_npz_sha256'))) == 1
                scaler = Path(rec['scaler_file'])
                rscaler = RELEASE/'partitions'/task/f'scaler_{fold}.json'
                row['scaler_byte_hash_matches_metadata_and_release'] = sha(scaler) == rec['scaler_sha256'] == sha(rscaler) == manifest[str(rscaler.relative_to(RELEASE))]
                row['scaler_split_array_hashes_match_task'] = read(scaler)['split_sha256'] == rec['split_sha256']
                assert row['npz_bytes_equal_to_release_and_zip']
                assert row['scaler_byte_hash_matches_metadata_and_release']
                assert row['scaler_split_array_hashes_match_task']
                rows.append(row)
    report = {
        'status': 'passed_no_partition_drift_type_mismatch_in_new_assertion',
        'cause': "TASK.fold.split_sha256 is a dictionary of canonical little-endian int64 array hashes, not an NPZ-file byte hash. Comparing sha(split_file), a string, to this dictionary must fail even when every input is unchanged.",
        'stale_task_hash_found': False,
        'actual_partition_drift_found': False,
        'tasks': 2, 'folds': 10, 'arrays': 40,
        'all_current_npz_bytes_equal_to_v3_release_tree_and_archive': True,
        'all_current_array_keys_dtypes_shapes_values_equal_to_v3_release': True,
        'all_array_hashes_and_counts_match_task_metadata': True,
        'all_task_json_hashes_match_original_data_protocol_and_release': True,
        'all_scalers_match_recorded_hashes_and_release': True,
        'recommended_pre_score_fix': "Verify {name: sha256(np.asarray(array,dtype='<i8').tobytes())} equals rec['split_sha256']; freeze the separate NPZ file SHA256 as an additional identity. Do not edit historical metadata or partition arrays.",
        'rows': rows, 'elapsed_seconds': time.time()-started,
        'affinity': sorted(os.sched_getaffinity(0)), 'GPU': False,
        'market_target_arrays_read': False, 'formal_stop_or_outer_scoring': False,
        'partition_files_changed': False, 'old_files_changed': False, 'Q3_values_read': False}
    (HERE/'INPUT_IDENTITY_REVIEW.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
