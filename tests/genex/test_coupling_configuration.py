from pathlib import Path

import pytest

from neutral_coupling.genex_coupling import genex_eirene_coupling as coupling


def write_config(path, extra=""):
    path.write_text(
        """
genex_path: ./genex
eirene_path: ./eirene
sum_temperatures: false
boundary_flux:
  factor_min: 0.25
  factor_max: 4.0
""" + extra,
        encoding="utf-8",
    )
    return path


def test_explicit_yaml_is_authoritative_and_resolves_paths(tmp_path):
    config = write_config(tmp_path / "settings.yaml")
    args = coupling.parse_coupling_args([
        "--config", str(config), "--SumTemp", "true",
        "--boundary-flux-factor-min", "0.9",
    ])
    assert args.SumTemp is False
    assert args.boundary_flux_factor_min == 0.25
    assert args.genex_path == tmp_path / "genex"
    assert args.filepattern == tmp_path / "input_sources"


def test_automatic_coupling_yaml_discovery(monkeypatch, tmp_path):
    write_config(tmp_path / "coupling.yaml")
    monkeypatch.chdir(tmp_path)
    args = coupling.parse_coupling_args([])
    assert args.SumTemp is False
    assert args.config == tmp_path / "coupling.yaml"


def test_cli_fallback_without_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = coupling.parse_coupling_args([
        "--pid", "123", "--genex_path", "/run", "--SumTemp", "false",
    ])
    assert args.pid == 123
    assert args.genex_path == "/run"
    assert args.SumTemp is False


@pytest.mark.parametrize("contents, match", [
    ("unknown: 1\n", "unknown coupling"),
    ("boundary_flux:\n  unknown: 1\n", "unknown boundary_flux"),
    ("sum_temperatures: nope\n", "wrong type"),
    ("[not yaml", "invalid YAML"),
])
def test_invalid_yaml_is_rejected(tmp_path, contents, match):
    config = tmp_path / "bad.yaml"
    config.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        coupling.load_coupling_config(config)


def test_environment_expansion_and_unset_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("COUPLING_TEST_RUN", str(tmp_path / "dynamic"))
    config = tmp_path / "settings.yaml"
    config.write_text("genex_path: ${COUPLING_TEST_RUN}\n", encoding="utf-8")
    assert coupling.load_coupling_config(config)["genex_path"] == tmp_path / "dynamic"

    config.write_text("genex_path: ${COUPLING_MISSING}\n", encoding="utf-8")
    monkeypatch.delenv("COUPLING_MISSING", raising=False)
    with pytest.raises(ValueError, match="unset environment"):
        coupling.load_coupling_config(config)


def test_pid_file_resolution_and_process_validation(monkeypatch, tmp_path):
    pid_file = tmp_path / "genex.pid"
    pid_file.write_text("4321\n", encoding="utf-8")
    args = coupling._config_namespace(coupling.load_coupling_config(
        write_config(tmp_path / "settings.yaml", "pid_file: ./genex.pid\n")
    ))
    monkeypatch.setattr(coupling.psutil, "pid_exists", lambda pid: pid == 4321)
    assert coupling.resolve_pid(args).pid == 4321

    monkeypatch.setattr(coupling.psutil, "pid_exists", lambda pid: False)
    with pytest.raises(ValueError, match="not running"):
        coupling.resolve_pid(args)


def test_complete_integration_examples_parse(monkeypatch):
    root = Path(__file__).parent / "integration" / "input_files"
    monkeypatch.setenv("GENEX_PATH", "/tmp/genex-integration-example")
    summed = coupling.coupling_args_from_file(root / "coupling_summed_temp.yaml")
    split = coupling.coupling_args_from_file(root / "coupling_multiple_temps.yaml")
    assert summed.SumTemp is True
    assert split.SumTemp is False
