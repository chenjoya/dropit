
import math
from fractions import Fraction, gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def common_integer(*numbers):
    fractions = [Fraction(n).limit_denominator() for n in numbers]
    multiple  = reduce(lcm, [f.denominator for f in fractions])
    ints      = [f * multiple for f in fractions]
    divisor   = reduce(gcd, ints)
    return [int(n / divisor) for n in ints]

def gcd(a, b, atol=1e-4):
    if (a < b):
        return gcd(b, a, atol=atol)
     
    # base case
    if (abs(b) < atol):
        return a
    else:
        return (gcd(b, a - math.floor(a / b) * b))

def coord(a, b):
    scale = a / b if a > b else b / a
    frac = scale
    error = [1.]
    shift = [0]
    idx = 0
    for i in range(32):
        tmp = frac * pow(2.0, i)
        cur = abs(round(tmp) - tmp)
        if cur < error[idx]:
            shift[idx] = i
            error[idx] = cur

    d = shift[idx]
    c = round(frac * pow(2.0, shift[idx]))
    print("c:{}, d: {}".format(c, d))
    scaled = c / pow(2., d)
    scaled = 1/ scaled if a > b else scaled
    print("scale {} vs scaled {}".format(scale, scaled))
    return scaled
    
    
if __name__ == "__main__":
    a, b = 1.3104, 1.8649
    print(common_integer(a, b))
    print(gcd(a, b))
    print(coord(a, b))

