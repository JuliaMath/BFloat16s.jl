using Test, BFloat16s

@testset "Comparisons" begin

    # DETERMINISTIC ROUNDING
    @test BFloat16(1)   <  BFloat16(2)
    @test BFloat16(1f0) <  BFloat16(2f0)
    @test BFloat16(1.0) <  BFloat16(2.0)
    @test BFloat16(1)   <= BFloat16(2)
    @test BFloat16(1f0) <= BFloat16(2f0)
    @test BFloat16(1.0) <= BFloat16(2.0)
    @test BFloat16(2)   >  BFloat16(1)
    @test BFloat16(2f0) >  BFloat16(1f0)
    @test BFloat16(2)   >= BFloat16(1)
    @test BFloat16(2f0) >= BFloat16(1f0)
    @test BFloat16(2.0) >= BFloat16(1.0)

    @test abs(BFloat16(-10)) == BFloat16(10)
    @test Float32(BFloat16(10)) == 1f1
    @test Float64(BFloat16(10)) == 10.0
    @test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
    @test sqrt(BFloat16(4f0)) == BFloat16(2f0)

    #STOCHASTIC ROUNDING
    @test BFloat16sr(1)   <  BFloat16sr(2)
    @test BFloat16sr(1f0) <  BFloat16sr(2f0)
    @test BFloat16sr(1.0) <  BFloat16sr(2.0)
    @test BFloat16sr(1)   <= BFloat16sr(2)
    @test BFloat16sr(1f0) <= BFloat16sr(2f0)
    @test BFloat16sr(1.0) <= BFloat16sr(2.0)
    @test BFloat16sr(2)   >  BFloat16sr(1)
    @test BFloat16sr(2f0) >  BFloat16sr(1f0)
    @test BFloat16sr(2)   >= BFloat16sr(1)
    @test BFloat16sr(2f0) >= BFloat16sr(1f0)
    @test BFloat16sr(2.0) >= BFloat16sr(1.0)

    # @test abs(BFloat16sr(-10)) == BFloat16sr(10)
    # @test Float32(BFloat16sr(10)) == 1f1
    # @test Float64(BFloat16sr(10)) == 10.0
    # @test BFloat16sr(2) ^ BFloat16sr(4) == BFloat16sr(16)
    # @test sqrt(BFloat16sr(4f0)) == BFloat16sr(2f0)

end
