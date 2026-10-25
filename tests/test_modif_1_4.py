import io

import numpy as np
import pytest

import laspy
from laspy import LazBackend
from tests.test_common import test1_4_las, write_then_read_again


@pytest.fixture(scope="session")
def las():
    return laspy.read(test1_4_las)


@pytest.fixture(params=LazBackend.detect_available())
def laz_backend(request):
    return request.param


def test_classification(las):
    las.classification[:] = 234
    assert np.all(las.classification == 234)

    res = write_then_read_again(las)

    assert np.all(las.classification == res.classification)


def test_intensity(las):
    las.intensity[:] = 89
    assert np.all(las.intensity == 89)
    res = write_then_read_again(las)

    assert np.all(las.intensity == res.intensity)


def test_writing_las_with_evlrs():
    las = laspy.read(test1_4_las)
    assert las.evlrs == []

    evlr = laspy.VLR(
        user_id="test",
        record_id=42,
        description="Just a test",
        record_data=b"And so he grinds his own hands",
    )
    las.evlrs.append(evlr)

    las_1 = write_then_read_again(las, do_compress=False)
    assert las_1.evlrs == [evlr]


def test_writing_laz_with_evlrs(laz_backend):
    las = laspy.read(test1_4_las)
    assert las.evlrs == []

    evlr = laspy.VLR(
        user_id="test",
        record_id=42,
        description="Just a test",
        record_data=b"And so he grinds he own hands",
    )
    las.evlrs.append(evlr)

    las_1 = write_then_read_again(las, do_compress=True, laz_backend=laz_backend)
    assert las_1.evlrs == [evlr]


def _stream_write_then_read_again(las, write_evlrs_call=None):
    stream = io.BytesIO()
    with laspy.open(
        stream, mode="w", closefd=False, header=las.header
    ) as writer:
        writer.write_points(las.points)
        if write_evlrs_call is not None:
            writer.write_evlrs(write_evlrs_call)
    stream.seek(0)
    return laspy.read(stream)


def test_streaming_writer_keeps_header_evlrs():
    # EVLRs on the header given to the streaming writer (the usual case
    # of a header copied from a reader, e.g. holding the WKT CRS) must
    # be written to the file, not silently dropped on close.
    las = laspy.read(test1_4_las)
    assert las.evlrs == []

    evlr = laspy.VLR(
        user_id="test",
        record_id=42,
        description="Just a test",
        record_data=b"And so he grinds his own hands",
    )
    las.evlrs.append(evlr)

    las_1 = _stream_write_then_read_again(las)
    assert las_1.evlrs == [evlr]
    assert las_1.header.number_of_evlrs == 1
    assert np.all(las_1.x == las.x)


def test_streaming_writer_keeps_wkt_crs_evlr():
    # The case from the wild: LAS 1.4 file whose CRS is stored as a
    # WKT EVLR; a streaming copy must keep both the EVLR and the
    # global_encoding.wkt flag that points at it.
    las = laspy.read(test1_4_las)
    las.header.global_encoding.wkt = 1
    wkt_evlr = laspy.vlrs.known.WktCoordinateSystemVlr()
    wkt_evlr.string = 'GEOGCS["WGS 84",DATUM["WGS_1984"]]'
    las.evlrs.append(wkt_evlr)

    las_1 = _stream_write_then_read_again(las)
    assert las_1.header.global_encoding.wkt == 1
    assert len(las_1.evlrs) == 1
    assert las_1.evlrs[0].record_data_bytes() == wkt_evlr.record_data_bytes()


def test_streaming_writer_does_not_duplicate_explicit_evlrs():
    # When the caller writes EVLRs explicitly, close() must not write
    # the header's EVLRs a second time.
    las = laspy.read(test1_4_las)
    header_evlr = laspy.VLR(
        user_id="test",
        record_id=42,
        description="On the header",
        record_data=b"header evlr",
    )
    las.evlrs.append(header_evlr)

    explicit_evlr = laspy.VLR(
        user_id="test",
        record_id=43,
        description="Written explicitly",
        record_data=b"explicit evlr",
    )
    las_1 = _stream_write_then_read_again(
        las, write_evlrs_call=laspy.vlrs.vlrlist.VLRList([explicit_evlr])
    )
    assert las_1.evlrs == [explicit_evlr]
