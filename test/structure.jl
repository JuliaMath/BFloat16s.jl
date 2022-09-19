half = BFloat16(0.5f0)
whole = BFloat16(1.0f0)
two = BFloat16(2.0f0)

phi = BFloat16(0.0f0 + Base.Math.MathConstants.golden)
invphi = BFloat16(1.0f0 / Base.Math.MathConstants.golden)
phi3 = phi * phi * phi
invphi3 = invphi * invphi * invphi

uint(x::BFloat16) = reinterpret(UInt16, x)

@testset "BFloat16 bits" begin
  @test uint(two) == 0x4000
  @test uint(half) == 0x3f00
  @test signbit(two) == false
end

@testset "BFloat16 parts" begin
  @test exponent(whole) == 0
  @test significand(whole) == zero(BFloat16)
  @test frexp(phi) == (BFloat16(0.30859375), 1)
  @test ldexp(BFloat16(0.30859375), 1) == phi
end


  
    
