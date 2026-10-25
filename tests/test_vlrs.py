import io
from pathlib import Path

import pytest

import laspy
from tests import test_common


def test_adding_classification_lookup():
    simple = laspy.read(test_common.simple_las)
    classification_lookup = laspy.vlrs.known.ClassificationLookupVlr()

    assert len(classification_lookup.lookups) == 0
    classification_lookup[20] = "computer"
    assert len(classification_lookup.lookups) == 1
    classification_lookup[17] = "car"

    simple.vlrs.append(classification_lookup)

    simple = test_common.write_then_read_again(simple)
    classification_lookups = simple.vlrs.get("ClassificationLookupVlr")[0]

    assert classification_lookups[20] == "computer"
    assert classification_lookups[17] == "car"


def test_lookup_out_of_range():
    classification_lookup = laspy.vlrs.known.ClassificationLookupVlr()
    with pytest.raises(ValueError):
        classification_lookup[541] = "LiquidWater"

    with pytest.raises(ValueError):
        classification_lookup[-42] = "SolidWater"


def test_classification_lookup_descriptions_are_not_mangled():
    # Descriptions in the wild contain characters other than
    # alphanumerics and spaces (underscores, hyphens, dots, commas);
    # parsing must keep them, only the NUL padding is stripped.
    record_data = b"".join(
        laspy.vlrs.known.ClassificationLookupVlr._lookup_struct.pack(
            class_id, description
        )
        for class_id, description in [
            (2, b"LEX_PotenialGND"),  # exactly 15 bytes, no padding
            (9, b"created, never"),
            (17, b"low-vegetation."),
            (42, b"wire - guard"),
        ]
    )
    classification_lookup = laspy.vlrs.known.ClassificationLookupVlr()
    classification_lookup.parse_record_data(record_data)

    assert classification_lookup[2] == "LEX_PotenialGND"
    assert classification_lookup[9] == "created, never"
    assert classification_lookup[17] == "low-vegetation."
    assert classification_lookup[42] == "wire - guard"

    # and writing the parsed VLR back gives the same bytes
    assert classification_lookup.record_data_bytes() == record_data


def test_classification_lookup_round_trip():
    classification_lookup = laspy.vlrs.known.ClassificationLookupVlr()
    classification_lookup[0] = "unclassified"
    classification_lookup[2] = "ground"
    classification_lookup[7] = "noise (low)"

    parsed_again = laspy.vlrs.known.ClassificationLookupVlr()
    parsed_again.parse_record_data(classification_lookup.record_data_bytes())

    assert parsed_again.lookups == classification_lookup.lookups
    assert parsed_again.record_data_bytes() == (
        classification_lookup.record_data_bytes()
    )


def test_adding_extra_bytes_vlr_by_hand():
    """
    Test that if someone adds an ExtraBytesVlr by himself
    without having matching extra bytes in the point record, the
    ExtraByteVlr is removed before writing
    """

    simple = laspy.read(test_common.simple_las)
    ebvlr = laspy.vlrs.known.ExtraBytesVlr()
    ebs = laspy.vlrs.known.ExtraBytesStruct(data_type=3, name="Fake".encode())
    ebvlr.extra_bytes_structs.append(ebs)
    simple.vlrs.append(ebvlr)
    assert len(simple.vlrs.get("ExtraBytesVlr")) == 1

    las = laspy.lib.write_then_read_again(simple)
    assert simple.points.point_size == las.points.point_size
    assert len(las.vlrs.get("ExtraBytesVlr")) == 0


def test_geokey_parsing_does_not_require_optional_params():
    las = laspy.read(str(Path(__file__).parent / "data/simple1_3.las"))
    geo_keys = laspy.vlrs.geotiff.parse_geo_tiff_keys_from_vlrs(las.vlrs)
    assert len(geo_keys) == 6


def test_cannot_write_vlrs_with_more_than_uint16_max_bytes():
    las = laspy.read(test_common.simple_las)
    big_junk_vlr = laspy.VLR(
        user_id="LASPY_ID",
        record_id=0,
        description="A VLR full of junk data",
        record_data=b"1" * (65_535 + 1),
    )
    las.vlrs.append(big_junk_vlr)

    with pytest.raises(ValueError):
        with io.BytesIO() as output:
            las.write(output)
