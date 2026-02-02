import pytest

# This file is deprecated - DataRSync class has been removed as data synchronization
# is no longer needed between client and server. All processing is now local.

@pytest.mark.skip(reason="DataRSync class has been removed")
def test_sync_removed():
    pass
