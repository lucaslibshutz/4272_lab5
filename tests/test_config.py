import numpy as np
import pytest

from bem_blade.config import load_config, operating_points, read_stations_file

from conftest import ROOT


def test_read_blade7_xlsx():
    d = read_stations_file(ROOT / "data" / "Blade7_geometry.xlsx")
    assert len(d["r"]) == len(d["chord"]) == len(d["theta_deg"]) == 20  # notes rows ignored
    assert d["r"][0] == pytest.approx(0.0254) and d["r"][-1] == pytest.approx(0.1778)
    assert d["theta_deg"][0] == pytest.approx(39.492, abs=1e-3)


def test_yaml12_floats_and_overrides(base_cfg, tmp_path):
    p = tmp_path / "c.yaml"
    import yaml
    text = yaml.safe_dump(base_cfg).replace("40000000000.0", "40.0e9")
    p.write_text(text)
    cfg = load_config(p, overrides=["operating.U_inf=[5, 7]", "rotor.pitch_deg=2.5"])
    assert cfg["structure"]["material"]["E"] == 40e9
    assert cfg["operating"]["U_inf"] == [5, 7] and cfg["rotor"]["pitch_deg"] == 2.5


def test_rpm_broadcast(base_cfg):
    base_cfg["operating"] = {"U_inf": 15.0, "rpm": [60.0, 120.0]}
    U, Om = operating_points(load_config(base_cfg))
    assert np.allclose(U, [15, 15]) and np.allclose(Om, [2 * np.pi, 4 * np.pi])


def test_rejects_unquoted_naca_and_bad_stations(base_cfg):
    bad = dict(base_cfg, airfoil={"naca": 12})
    with pytest.raises(ValueError, match="quoted"):
        load_config(bad)
    base_cfg["stations"]["r"] = base_cfg["stations"]["r"][::-1]
    with pytest.raises(ValueError, match="increasing"):
        load_config(base_cfg)
