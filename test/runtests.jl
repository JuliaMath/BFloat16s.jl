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

N = 10000

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

@testset "1+eps/4 is round 25% up" begin

    p1 = 0
    p2 = 0

    eps = 0.0078125f0
    x = 1 + eps/4

    for i = 1:N
        f = Float32(BFloat16_stochastic_round(x))
        if 1.0f0 == f
            p1 += 1
        elseif 1 + eps == f
            p2 += 1
        end
    end

    @test p1+p2 == N
    @test p1/N > 0.70
    @test p1/N < 0.80
end

@testset "2+eps/4 is round 25% up" begin

    p1 = 0
    p2 = 0

    eps = 0.0078125f0
    x = 2 + eps/2

    for i = 1:N
        f = Float32(BFloat16_stochastic_round(x))
        if 2.0f0 == f
            p1 += 1
        elseif 2 + 2eps == f
            p2 += 1
        end
    end

    @test p1+p2 == N
    @test p1/N > 0.70
    @test p1/N < 0.80
end

@testset "powers of 2 are not round" begin

    p1 = 0
    p2 = 0

    for x in Float32[2,4,8,16,32,64,128,256,512,1024]
        for i = 1:100
            @test x == Float32(BFloat16_stochastic_round(x))
            @test x == Float32(BFloat16(x))
        end
    end

    for x in Float32[1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256,1/512,1/1024]
        for i = 1:100
            @test x == Float32(BFloat16_stochastic_round(x))
            @test x == Float32(BFloat16(x))
        end
    end
end

@testset "1+eps+eps/8 is round 12.5% up" begin

    p1 = 0
    p2 = 0
    N = 1000000

    eps = 0.0078125f0
    x = 1 + eps + eps/8

    for i = 1:N
        f = Float32(BFloat16_stochastic_round(x))
        if 1.0f0 + eps == f
            p1 += 1
        elseif 1 + 2eps == f
            p2 += 1
        end
    end
    println((p1/N,p2/N))
    @test p1+p2 == N
    @test p1/N > 0.85
    @test p1/N < 0.90
end

@testset "1+eps/8 is round 12.5% up" begin

    p1 = 0
    p2 = 0
    N = 1000000

    eps = 0.0078125f0
    x = 1 + eps/8

    for i = 1:N
        f = Float32(BFloat16_stochastic_round(x))
        if 1.0f0 == f
            p1 += 1
        elseif 1 + eps == f
            p2 += 1
        end
    end
    println((p1/N,p2/N))
    @test p1+p2 == N
    @test p1/N > 0.85
    @test p1/N < 0.90
end

@testset "1+eps/16 is round 6.25% up" begin

    p1 = 0
    p2 = 0
    N = 1000000

    eps = 0.0078125f0
    x = 1 + eps/16

    for i = 1:N
        f = Float32(BFloat16_stochastic_round(x))
        if 1.0f0 == f
            p1 += 1
        elseif 1 + eps == f
            p2 += 1
        end
    end
    println((p1/N,p2/N))
    @test p1+p2 == N
    @test p2/N > 0.055
    @test p2/N < 0.07
end

@testset "2+eps/16 is round 6.25% up" begin

    p1 = 0
    p2 = 0
    N = 1000000

    eps = 0.0078125f0
    x = 2 + eps/8

    for i = 1:N
        f = Float32(BFloat16_stochastic_round(x))
        if 2.0f0 == f
            p1 += 1
        elseif 2 + 2eps == f
            p2 += 1
        end
    end
    println((p1/N,p2/N))
    @test p1+p2 == N
    @test p2/N > 0.055
    @test p2/N < 0.07
end

@testset "Subnormals are deterministically round" begin

    for hex in 0x1:0x80     # 0x80 == 0x1 << 7  # test for all subnormals of BFloat16

        x = reinterpret(Float32,UInt32(hex) << 16)

        for i = 1:10
            @test x == Float32(BFloat16(x))
            @test x == Float32(BFloat16sr(x))
            @test x == Float32(BFloat16_stochastic_round(x))
        end
    end
end




@test abs(BFloat16(-10)) == BFloat16(10)
@test Float32(BFloat16(10)) == 1f1
@test Float64(BFloat16(10)) == 10.0
@test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
@test sqrt(BFloat16(4f0)) == BFloat16(2f0)
