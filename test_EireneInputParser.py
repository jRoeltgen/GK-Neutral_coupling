import math
from pathlib import Path

import pytest
import scipy.constants as pyconst
from EireneInputParser import EireneInputParser, ValidationReport


INPUT_FILE = Path(__file__).parent / "test_data" / "eirene_data" / "input.dat"


def parser_for_mass_validation(species, masses):
    parser = EireneInputParser.__new__(EireneInputParser)
    parser.requested_strata = [1]
    parser.species = species
    parser.masses = masses
    parser.volume_recombination = {"particle": {}, "energy": {}}
    parser.report = ValidationReport()
    return parser


def test_full_input_parses_and_validates_species_masses():
    parser = EireneInputParser(INPUT_FILE)

    result = parser.parse_all()

    assert result["validation"] is parser.report
    assert parser.report.errors == []

    assert parser.species == {
        "atoms": ["D", "T", "ATOMS"],
        "molecules": ["D2", "T2", "MOLECULES"],
        "test_ions": ["D2+", "T2+", "TEST IONS"],
        "bulk_ions": ["D+", "T+"],
        "electrons": ["ELECTRONS"],
    }

    expected_mass_numbers = {
        "atoms": [2, 3],
        "molecules": [4, 6],
        "test_ions": [4, 6],
        "bulk_ions": [2, 3],
    }
    for particle_class, mass_numbers in expected_mass_numbers.items():
        assert parser.masses[particle_class] == pytest.approx(
            [number * pyconst.proton_mass for number in mass_numbers],
            rel=1e-12,
            abs=0.0,
        )
        assert all(
            "(B)" not in species
            for species in parser.species[particle_class]
        )

    assert parser.masses["electrons"] == [pyconst.electron_mass]
    assert len(parser.masses["bulk_ions"]) == len(parser.species["bulk_ions"])


def test_mass_validation_reports_count_mismatch_and_invalid_mass():
    parser = parser_for_mass_validation(
        {"atoms": ["D", "T"], "electrons": ["ELECTRONS"]},
        {"atoms": [math.nan], "electrons": [1.0]},
    )

    parser.validate_schema()

    assert any("Mass count mismatch" in error for error in parser.report.errors)
    assert any("Invalid mass for 'D'" in error for error in parser.report.errors)
