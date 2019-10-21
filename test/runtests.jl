using Test, BFloat16s

@test BFloat16(1)   < BFloat16(2)
@test BFloat16(1f0) < BFloat16(2f0)
@test BFloat16(1.0) < BFloat16(2.0)
@test abs(BFloat16(-10)) == BFloat16(10)
@test Float32(BFloat16(10)) == 1f1
@test Float64(BFloat16(10)) == 10.0
@test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
