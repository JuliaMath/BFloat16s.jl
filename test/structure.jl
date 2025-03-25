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
    @test significand(whole) == one(BFloat16)

    @testset "subnormal" begin
        allbfs = UInt16(1):typemax(UInt16)
        res = map(allbfs) do raw
            bf = reinterpret(BFloat16, raw)
            issubnormal(bf) == issubnormal(Float32(bf))
        end
        @test all(res)
        @test significand(reinterpret(BFloat16, 0b0000000000000001)) == one(BFloat16)
    end

    @test frexp(phi) == (BFloat16(0.80859375), 1)
    @test ldexp(BFloat16(0.80859375), 1) == phi

    @test exponent(invphi3) == -3
    @test significand(invphi3) == BFloat16(1.8828125)

    fr,xp = frexp(invphi3)
    @test xp == -2
    @test fr == BFloat16(0.94140625)
    @test ldexp(fr, xp) == invphi3
end
