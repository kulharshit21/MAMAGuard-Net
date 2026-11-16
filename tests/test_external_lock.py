from pathlib import Path

import pytest

from mamaguardnet.data.external_lock import (
    ExternalTestLocked,
    assert_external_allowed,
    consume_external,
    freeze_external,
)


def test_external_token_is_hash_bound_and_single_use(tmp_path: Path):
    state = tmp_path / "external.json"
    with pytest.raises(ExternalTestLocked):
        assert_external_allowed(state, config_hash="c", checkpoint_hash="k")
    freeze_external(state, config_hash="c", checkpoint_hash="k")
    assert_external_allowed(state, config_hash="c", checkpoint_hash="k")
    with pytest.raises(ExternalTestLocked):
        assert_external_allowed(state, config_hash="wrong", checkpoint_hash="k")
    consume_external(state, config_hash="c", checkpoint_hash="k")
    with pytest.raises(ExternalTestLocked):
        assert_external_allowed(state, config_hash="c", checkpoint_hash="k")

