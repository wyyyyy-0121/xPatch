"""Inspect an actual toy return archive; protect data and preserve raw evidence."""
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from .common import OUT,write_json,digest
from . import cloud,accounting,restore


def test_return_archive_preserves_predictions_and_weights_but_excludes_market_data(monkeypatch):
    with TemporaryDirectory(prefix='unit_return_',dir=OUT) as temporary:
        root=Path(temporary);outputs=root/'outputs/fusion_next';outputs.mkdir(parents=True)
        monkeypatch.setattr(cloud,'ROOT',root);monkeypatch.setattr(cloud,'OUT',outputs)
        monkeypatch.setenv('FUSION_EXECUTION_PROFILE','autodl')
        monkeypatch.setattr(accounting,'totals',lambda:{'active_jobs':[]})
        write_json(outputs/'confirmation.json',{'evaluated':False})
        for name in ('deployment/fixed/model.pt','trials/config/prediction.npz',
                     'raw/AAPL/page.json','final_test/raw/AAPL/page.json','trials/.env'):
            path=outputs/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'toy-only')
        receipt=cloud.return_bundle()
        with tarfile.open(root/receipt['archive']) as archive:
            names=set(archive.getnames());manifest=json.load(archive.extractfile('outputs/fusion_next/cloud/return_manifest.json'))
        assert 'outputs/fusion_next/deployment/fixed/model.pt' in names
        assert 'outputs/fusion_next/trials/config/prediction.npz' in names
        assert not any('/raw/' in name or '/.env' in name for name in names)
        assert manifest['Q3_data_included'] is False and manifest['credentials_included'] is False


def test_return_restore_checks_every_file_before_install_and_preserves_local_timings(monkeypatch):
    import pytest
    with TemporaryDirectory(prefix='unit_restore_',dir=OUT) as temporary:
        root=Path(temporary);remote=root/'remote';local=root/'local'
        remote_out=remote/'outputs/fusion_next';local_out=local/'outputs/fusion_next'
        for directory in (remote_out,local_out,local/'tmp'):directory.mkdir(parents=True)
        monkeypatch.setattr(cloud,'ROOT',remote);monkeypatch.setattr(cloud,'OUT',remote_out)
        monkeypatch.setenv('FUSION_EXECUTION_PROFILE','autodl')
        monkeypatch.setattr(accounting,'totals',lambda:{'active_jobs':[]})
        write_json(remote_out/'confirmation.json',{'evaluated':False,'unit_test':True})
        write_json(remote_out/'performance_cpu.json',{'machine':'cloud','unit_test':True})
        fitted=remote_out/'deployment/fixed/model.pt';fitted.parent.mkdir(parents=True);fitted.write_bytes(b'toy fitted parameters')
        bundle=cloud.return_bundle();archive=remote/bundle['archive']
        monkeypatch.setenv('FUSION_EXECUTION_PROFILE','local')
        monkeypatch.setattr(restore,'ROOT',local);monkeypatch.setattr(restore,'OUT',local_out)
        write_json(local_out/'performance_cpu.json',{'machine':'local','unit_test':True})
        existing=local_out/'deployment/fixed/model.pt';existing.parent.mkdir(parents=True);existing.write_bytes(b'different existing toy fit')
        with pytest.raises(ValueError,match='nothing restored'):restore.restore(archive,bundle['sha256'])
        assert not (local_out/'confirmation.json').exists() and existing.read_bytes()==b'different existing toy fit'
        existing.unlink()
        with pytest.raises(ValueError,match='separate SHA'):restore.restore(archive,'0'*64)
        tampered=root/'tampered.tar'
        import io
        with tarfile.open(archive) as original,tarfile.open(tampered,'w') as changed:
            for member in original.getmembers():
                body=original.extractfile(member).read()
                if member.name=='outputs/fusion_next/deployment/fixed/model.pt':body=b'x'*len(body)
                changed.addfile(member,io.BytesIO(body))
        with pytest.raises(ValueError,match='file identity'):restore.restore(tampered,digest(tampered))
        assert not (local_out/'confirmation.json').exists() and not existing.exists()
        installed=restore.restore(archive,bundle['sha256'])
        assert installed['restored_files']==3 and len(installed['preserved_local_diagnostics'])==1
        assert json.loads((local_out/'performance_cpu.json').read_text())['machine']=='cloud'
        assert json.loads((local/installed['preserved_local_diagnostics'][0]).read_text())['machine']=='local'
        assert digest(existing)==digest(fitted)
        again=restore.restore(archive,bundle['sha256'])
        assert again['restored_files']==0 and again['already_identical_files']==3


def test_return_restore_rejects_tampered_and_unlisted_payloads_before_install(monkeypatch):
    import io
    import pytest
    with TemporaryDirectory(prefix='unit_restore_bad_',dir=OUT) as temporary:
        root=Path(temporary);output=root/'outputs/fusion_next';output.mkdir(parents=True);(root/'tmp').mkdir()
        monkeypatch.setattr(restore,'ROOT',root);monkeypatch.setattr(restore,'OUT',output)
        monkeypatch.setenv('FUSION_EXECUTION_PROFILE','local')
        # Even a matching archive SHA must not authorize undeclared data/paths.
        for name in ('outputs/fusion_next/raw/AAPL/page.json','../escape.json'):
            archive=root/'bad.tar'
            with tarfile.open(archive,'w') as stream:
                member=tarfile.TarInfo(name);member.size=2;stream.addfile(member,io.BytesIO(b'{}'))
            with pytest.raises(ValueError,match='Unexpected'):restore.restore(archive,digest(archive))
        assert not (root/'escape.json').exists() and not (output/'raw').exists()
