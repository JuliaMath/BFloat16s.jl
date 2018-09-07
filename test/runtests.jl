using BFloat16s
using Test

bf1 = BFloat16(1.0)
bf2 = BFloat16(2.0)
bf3 = BFloat16(3.0)
bf4 = BFloat16(4.0)

@test bf1 + bf2 == bf3
@test bf2 - bf1 == bf1
@test bf2 * bf2 == bf4
@test bf4 / bf2 == bf2
