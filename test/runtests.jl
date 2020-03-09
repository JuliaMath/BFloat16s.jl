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
end

N = 1000

@testset "Rounding" begin

    @testset "1.0 always to 1.0" begin
        for i = 1:N
            @test 1.0f0 == Float32(BFloat16(1.0f0))
            @test 1.0f0 == Float32(BFloat16sr(1.0f0))
            @test 1.0f0 == Float32(BFloat16_stochastic_round(1.0f0))
        end
    end

    @testset "1+eps/2 is round 50/50 up/down" begin

        p1 = 0
        p2 = 0

        eps = 0.0078125f0
        x = 1 + eps/2

        for i = 1:N
            f = Float32(BFloat16_stochastic_round(x))
            if 1.0f0 == f
                p1 += 1
            elseif 1 + eps == f
                p2 += 1
            end
        end

        @test p1+p2 == N
        @test p1/N > 0.45
        @test p1/N < 0.55
    end

    @testset "Subnormals are deterministically round" begin

        for hex in 0x1:0x80     # 0x80 == 0x1 << 7  # test for all subnormals of BFloat16

            x = reinterpret(Float32,UInt32(hex) << 16)

            for i = 1:N
                @test x == Float32(BFloat16(x))
                @test x == Float32(BFloat16sr(x))
                @test x == Float32(BFloat16_stochastic_round(x))
            end
        end
    end
end




@test abs(BFloat16(-10)) == BFloat16(10)
@test Float32(BFloat16(10)) == 1f1
@test Float64(BFloat16(10)) == 10.0
@test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
@test sqrt(BFloat16(4f0)) == BFloat16(2f0)
