from refactorDecimalsToFractions import get_gcd, get_numerator_denominator


def test_get_gcd():
    assert get_gcd(9, 20) == 1
    assert get_gcd(45, 100) == 5
    assert get_gcd(1, 1) == 1
    assert get_gcd(0, 9) == 0
    assert get_gcd(0, 1) == 0
    assert get_gcd(0, 0) is None


def test_get_numerator_denominator():
    assert get_numerator_denominator("0.45") == (45, 100, False)
    assert get_numerator_denominator(0.45) == (45, 100, False)
    assert get_numerator_denominator("0.4500") == (45, 100, False)
    assert get_numerator_denominator(0.45000) == (45, 100, False)
    assert get_numerator_denominator(0) == (0, 1, False)
    assert get_numerator_denominator("0") == (0, 1, False)
    assert get_numerator_denominator("-0") == (0, 1, False)
    assert get_numerator_denominator("12.5") == (125, 10, False)
    assert get_numerator_denominator(12.50) == (125, 10, False)
    assert get_numerator_denominator(-12.50) == (125, 10, True)
    assert get_numerator_denominator("a") is None
    assert get_numerator_denominator("") is None
    assert get_numerator_denominator(1.0) == (1, 1, False)
    assert get_numerator_denominator(10.0) == (10, 1, False)
    assert get_numerator_denominator(-1.0) == (1, 1, True)
    assert get_numerator_denominator(-10.0) == (10, 1, True)
