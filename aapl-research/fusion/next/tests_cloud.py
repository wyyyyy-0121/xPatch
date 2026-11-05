"""Inspect an actual toy return archive; protect data and preserve raw evidence."""
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from .common import OUT,write_json
from . import cloud,accounting


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
