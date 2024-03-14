import pytest

from dcp_client.utils.sync_src_dst import DataRSync


@pytest.fixture
def rsyncer():
    syncer = DataRSync(user_name="local", host_name="local", server_repo_path=".")
    return syncer


def test_init(rsyncer):
    assert rsyncer.user_name == "local"
    assert rsyncer.host_name == "local"
    assert rsyncer.server_repo_path == "."


def test_first_sync_e(rsyncer):
    msg, _ = rsyncer.first_sync("eval_data_path")
    assert msg == "Error"


def test_sync(rsyncer):
    msg, _ = rsyncer.sync("server", "client", "eval_data_path")
    assert msg == "Error"
