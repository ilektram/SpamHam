

def get_gcd(a, b):
    if a == 0 and b == 0:
        return None
    elif a == 0 or b == 0:
        return 0
    common_denominators_l = []
    minAB = min(a, b)
    for i in range(1, minAB + 1):
        if a % i == 0 and b % i == 0:
            common_denominators_l.append(i)
    return max(common_denominators_l)


def get_numerator_denominator(n):
    try:
        new_n = float(n)
    except ValueError:
        print("Please provide a number next time...")
        return None
    if new_n == 0:
        return (0, 1)
    num, dec = str(n).split(".")
    stripped_dec = dec.rstrip("0")
    denominator = int(10 ** len(stripped_dec))
    numerator = int(new_n * denominator)
    return (numerator, denominator)


def convert_to_fraction_string(n):
    print(f"Received {str(n)} as input...")
    numerator, denominator = get_numerator_denominator(n)
    gcd = get_gcd(numerator, denominator)
    if gcd:
        res = "{}/{}".format(int(numerator/gcd), int(denominator/gcd))
        return res
    else:
        return None


if __name__ == "__main__":
    n = input('Enter a decimal number please: ')
    result = convert_to_fraction_string(n)
    if result:
        print("{} can be written as {}".format(n, result))
    else:
        print("Fraction not found")
