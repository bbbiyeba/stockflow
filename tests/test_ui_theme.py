import math

from src.ui_theme import money, pct, units


def test_money_formats():
    assert money(1_234_567) == "$1.2M"
    assert money(340_000) == "$340k"
    assert money(1_500) == "$1.5k"
    assert money(950) == "$950"
    assert money(-2_500_000) == "-$2.5M"
    assert money(math.nan) == "–"


def test_pct_and_units():
    assert pct(0.043) == "4.3%"
    assert pct(0.5, 0) == "50%"
    assert units(1234) == "1,234"
    assert units(250_000) == "250k"
    assert units(None) == "–"
