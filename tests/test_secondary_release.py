import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('unpack_secondary_release',Path(__file__).resolve().parents[1]/'scripts/unpack_secondary_release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('badname',['../escape','/absolute','a\\escape','C:escape'])
def test_release_rejects_unsafe_paths(tmp_path,badname):
    archive = tmp_path/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info = tarfile.TarInfo(badname)
        info.size = 1
        tar.addfile(info,io.BytesIO(b'x'))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'archive':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'file_count':1}))
    with pytest.raises(ValueError,match='Unsafe'):
        module.unpack(manifest,tmp_path/'destination')
    assert not (tmp_path/'destination').exists()
