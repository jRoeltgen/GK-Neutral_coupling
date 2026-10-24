from pathlib import Path

import numpy as np

from neutral_coupling.common.b2_io import B2
from neutral_coupling.common.eirene_io import eirene
from neutral_coupling.genex_coupling.eirene_interface import prepare_fort31
from neutral_coupling.utility.create_fort31 import build_fort31, create_fort31


DATA = Path(__file__).parents[1] / "genex" / "integration" / "input_files"
SOLPS_REFERENCE = (
    Path(__file__).parents[1]
    / "genex"
    / "reference_data"
    / "fort.31.solps_reference"
)


def test_build_fort31_geometry_and_masks():
    geometry = B2(DATA).gmtry
    assert geometry["leftix"].shape == geometry["vol"].shape

    generated = build_fort31(
        DATA / "input.dat", DATA / "b2fgmtry", DATA / "fort.30"
    )
    reference = eirene()
    reference.read_ft31(SOLPS_REFERENCE, 192, 38, 1)

    np.testing.assert_allclose(generated.fort31["vol"], reference.fort31["vol"])
    np.testing.assert_allclose(generated.fort31["bb"], reference.fort31["bb"])
    np.testing.assert_allclose(
        generated.fort31["pitch_angle"], reference.fort31["pitch_angle"]
    )

    # Final face planes are not represented in B2 and must remain zero.
    assert np.all(generated.fort31["fnax"][-1, :] == 0.0)
    assert np.all(generated.fort31["fnay"][:, -1] == 0.0)
    assert np.count_nonzero(generated.fort31["fnax"]) > 0
    assert np.count_nonzero(generated.fort31["fnay"]) > 0


def test_written_file_round_trips(tmp_path):
    output = tmp_path / "fort.31"
    create_fort31(
        DATA / "input.dat", DATA / "b2fgmtry", DATA / "fort.30", output
    )
    reread = eirene()
    reread.read_ft31(output, 192, 38, 1)
    assert reread.fort31["na"].shape == (192, 38)
    assert np.all(reread.fort31["ion_charge"] == 1.0)


def test_generated_and_stored_skeletons_prepare_identically(tmp_path):
    """Generated and SOLPS skeletons prepare identically except sheath data."""
    nx, ny = 192, 38
    generated_path = tmp_path / "fort.31"
    create_fort31(
        DATA / "input.dat",
        DATA / "b2fgmtry",
        DATA / "fort.30",
        generated_path,
    )

    stored = eirene()
    stored.read_ft31(SOLPS_REFERENCE, nx, ny, 1)
    generated = eirene()
    generated.read_ft31(generated_path, nx, ny, 1)

    x = np.arange(nx, dtype=float)[:, None]
    y = np.arange(ny, dtype=float)[None, :]
    field = 1.0 + 0.01 * x + 0.001 * y
    fake_data = {
        "n": {"ions": field},
        "u_par": {"ions": field + 1.0},
        "u_rad": {"ions": field + 2.0},
        "u_phi": {"ions": field + 3.0},
        "Ttot": {"electrons": field + 4.0, "ions": field + 5.0},
        "es_pot": {"scalar": field + 6.0},
        "pr": {"scalar": field + 7.0},
        "fnax": {"ions": field + 8.0},
        "fnay": {"ions": field + 9.0},
        "Q_par": {"electrons": field + 10.0, "ions": field + 11.0},
    }

    prepare_fort31(stored, fake_data, ["electrons"], ["ions"])
    prepare_fort31(generated, fake_data, ["electrons"], ["ions"])

    stored_path = tmp_path / "stored.prepared.31"
    generated_path = tmp_path / "generated.prepared.31"
    stored.write_ft31(stored_path)
    generated.write_ft31(generated_path)
    stored_roundtrip = eirene()
    generated_roundtrip = eirene()
    stored_roundtrip.read_ft31(stored_path, nx, ny, 1)
    generated_roundtrip.read_ft31(generated_path, nx, ny, 1)

    assert generated_roundtrip.fort31.keys() == stored_roundtrip.fort31.keys()
    for field_name in stored_roundtrip.fort31:
        np.testing.assert_allclose(
            generated_roundtrip.fort31[field_name],
            stored_roundtrip.fort31[field_name],
            rtol=1.0e-7,
            atol=1.0e-12,
            err_msg=f"fort.31 field differs: {field_name}",
        )
