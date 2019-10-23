using Test, BFloat16s

@testset "comparisons" begin
	@test BFloat16(1)   <  BFloat16(2)
	@test BFloat16(1f0) <  BFloat16(2f0)
	@test BFloat16(1.0) <  BFloat16(2.0)
	@test BFloat16(1)   <= BFloat16(2)
	@test BFloat16(1f0) <= BFloat16(2f0)
	@test BFloat16(1.0) <= BFloat16(2.0)
	@test BFloat16(2)   >  BFloat16(1)
	@test BFloat16(2f0) >  BFloat16(1f0)
	@test BFloat16(2.0) >  BFloat16(1.0)
	@test BFloat16(2)   >= BFloat16(1)
	@test BFloat16(2f0) >= BFloat16(1f0)
	@test BFloat16(2.0) >= BFloat16(1.0)
	@test BFloat16(2)   != BFloat16(1)
	@test BFloat16(2f0) != BFloat16(1f0)
	@test BFloat16(2.0) != BFloat16(1.0)
end

@testset "conversions" begin
	@test Float32(BFloat16(10)) == 1f1
	@test Float64(BFloat16(10)) == 10.0
end

@testset "functions" begin
	@test abs(BFloat16(-10)) == BFloat16(10)
	@test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
	@test eps(BFloat16) == BFloat16(0.0078125)
	@test sqrt(BFloat16(4f0)) == BFloat16(2f0)
end
