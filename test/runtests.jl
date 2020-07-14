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

@testset "arithmetic" begin
  @test BFloat16(0.2) * BFloat16(5.0) == BFloat16(1.0)
  @test BFloat16(1.0f0) / BFloat16(5.0f0) == BFloat16(0.2f0)
  @test inv(BFloat16(5.0)) == BFloat16(0.2)
  @test zero(BFloat16) == BFloat16(0.0f0)
  @test one(BFloat16) == BFloat16(1.0)
end

@testset "conversions" begin
	@test Float32(BFloat16(10)) == 1f1
	@test Float64(BFloat16(10)) == 10.0
	@test Int32(BFloat16(10)) == Int32(10)
	@test Int64(BFloat16(10)) == Int64(10)
end

@testset "functions" begin
	@test abs(BFloat16(-10)) == BFloat16(10)
	@test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
	@test eps(BFloat16) == BFloat16(0.0078125)
	@test sqrt(BFloat16(4f0)) == BFloat16(2f0)
	@test round(BFloat16(10.4), RoundUp) == BFloat16(11.0)
	@test round(BFloat16(10.6), RoundDown) == BFloat16(10.0)
	@test round(BFloat16(3.2), RoundNearest) == BFloat16(3.0)
	@test round(BFloat16(4.8), RoundNearest) == BFloat16(5.0)
end
