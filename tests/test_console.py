from bem_blade import load_config, run, summary_text
from bem_blade.console import eng

from conftest import ROOT


def test_eng_prefixes():
    assert eng(0.414, "N·m") == "414 mN·m"
    assert eng(1.994e6, "N·m") == "1.99 MN·m"
    assert eng(-4.89e6, "Pa") == "-4.89 MPa"


def test_summary_text_sweep():
    res = run(load_config(ROOT / "examples" / "blade7.yaml",
                          overrides=["operating.rpm=[2000, 3600, 6000]"]))
    txt = summary_text(res, "blade7")
    for s in ("MAX MOMENTS", "M_flap", "M_edge", "STRENGTH", "OPERATING POINTS", "6000 RPM",
              "best C_P"):
        assert s in txt
    assert "\x1b[" not in txt  # plain text, no ANSI codes


def test_summary_text_single_point(base_cfg):
    txt = summary_text(run(load_config(base_cfg)))
    assert "MAX MOMENTS" in txt and "Performance" in txt and "OPERATING POINTS" not in txt
