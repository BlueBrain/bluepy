import os

import pytest

from bluepy.connectome import Connectome
from bluepy.exceptions import BluePyError
from bluepy.impl.connectome_nrn import NrnConnectome
from bluepy.impl.connectome_sonata import SonataConnectome

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_connectome():
    nrn = Connectome(os.path.join(TEST_DATA_DIR, 'nrn/nrn.h5'), {})
    assert isinstance(nrn._impl, NrnConnectome)

    sonata = Connectome(os.path.join(TEST_DATA_DIR, 'edges.sonata'), {})
    assert isinstance(sonata._impl, SonataConnectome)

    sonata_h5 = Connectome(os.path.join(TEST_DATA_DIR, 'edges.h5'), {})
    assert isinstance(sonata_h5._impl, SonataConnectome)

    with pytest.raises(BluePyError):
        Connectome('dummy.syn2', {})

    with pytest.raises(BluePyError):
        Connectome(os.path.join(TEST_DATA_DIR, 'B.target'), {})
