import numpy as np
import debug
import pytest


@pytest.fixture()
def logger(tmpdir):
    path = tmpdir.join('out.zarr')
    return debug.ZarrLogger(str(path))


def test_append(logger):
    key = 'U'
    arr = np.random.rand(100, 10)

    logger.append(key, arr)
    logger.append(key, arr)

    assert logger.root[key].shape == (2, ) + arr.shape

    with pytest.raises(ValueError):
        logger.append(key, np.random.rand(10, 10))

    # test scalar appending
    key = 'time'
    logger.append(key, np.array([1.0]))
    logger.append(key, np.array([1.0]))


def test_set(logger):
    key = 'U'
    arr = np.random.rand(100, 10)
    logger.set(key, arr)

    a = logger.get(key)[:]
    np.testing.assert_equal(a, arr)


def test_tarfolder(tmpdir):
    import io
    import tarfile
    path = tmpdir.join('a')
    with open(path, "w") as f:
        f.write("hello world")

    # get tarfile as string
    s = debug.get_tar_data(tmpdir)
    out_path = tmpdir.mkdir('out')
    debug.extract_tar_data(s, str(out_path))

    # assert that "a" contains "hello world"
    with open(out_path.join("a")) as f:
        assert f.read() == "hello world"
