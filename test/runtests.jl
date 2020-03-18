using Test, BFloat16s

@testset "Comparisons" begin
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
    @test InfB16 > -InfB16
    @test NaNB16 != NaNB16
end

@testset "Sign flip" begin
    @test one(BFloat16) == -(-(one(BFloat16)))
    @test zero(BFloat16) == -(zero(BFloat16))
end

@testset "Integer promotion" begin
    f = BFloat16(1)
    @test 2f == BFloat16(2)
    @test 0 == BFloat16(0)
end

@testset "Rounding" begin
    @test 1 == Int(round(BFloat16(1.2)))
    @test 1 == Int(floor(BFloat16(1.2)))
    @test 2 == Int(ceil(BFloat16(1.2)))

    @test -1 == Int(round(BFloat16(-1.2)))
    @test -2 == Int(floor(BFloat16(-1.2)))
    @test -1 == Int(ceil(BFloat16(-1.2)))
end

@testset "Nextfloat prevfloat" begin
    o = one(BFloat16)
    @test o == nextfloat(prevfloat(o))
    @test o == prevfloat(nextfloat(o))
end

@testset "Absolute values" begin
    @test abs(BFloat16(-10)) == BFloat16(10)
    @test abs(BFloat16(-0)) == BFloat16(0)
    @test abs(BFloat16(1)) == BFloat16(1)
end

@testset "Conversion float" begin
    fs = [1.0,2.5,10.0,0.0,-0.25,-5.0]

    for f in fs
        @test f == Float64(BFloat16(f))
        @test Float32(f) == Float32(BFloat16(f))
        @test Float16(f) == Float16(BFloat16(f))
    end
end

@testset "Conversion int" begin
    fs = [-5,-1,0,-0,1,2]

    for f in fs
        @test f == Int64(BFloat16(f))
        @test Int32(f) == Int32(BFloat16(f))
        @test Int16(f) == Int16(BFloat16(f))
    end
end

@testset "Addition and subtraction" begin

    @test BFloat16(2) == BFloat16(1)+BFloat16(1)
    @test BFloat16(1) == BFloat16(1)+BFloat16(0)
    @test BFloat16(-1) == BFloat16(1)+BFloat16(-2)
    @test BFloat16(1.5) == BFloat16(1)+BFloat16(0.5)
    @test BFloat16(1.5) == BFloat16(2)-BFloat16(0.5)
    @test BFloat16(0) == BFloat16(1.2345)-BFloat16(1.2345)
    @test BFloat16(-1.5) == BFloat16(2.5)-BFloat16(4.0)
end

@testset "Multiplication" begin

    @test BFloat16(2) == BFloat16(2)*BFloat16(1)
    @test BFloat16(0) == BFloat16(1)*BFloat16(0)
    @test BFloat16(-2) == BFloat16(1)*BFloat16(-2)
    @test BFloat16(0.25) == BFloat16(0.5)*BFloat16(0.5)
    @test BFloat16(1.5) == BFloat16(3)*BFloat16(0.5)
    @test BFloat16(12.5) == BFloat16(1.25)*BFloat16(10.0)
    @test BFloat16(-12) == BFloat16(-10)*BFloat16(1.2)
    @test InfB16 == -InfB16*-InfB16
end

@testset "Division" begin

    @test BFloat16(2) == BFloat16(2)/BFloat16(1)
    @test BFloat16(0) == BFloat16(0)/BFloat16(1)
    @test BFloat16(-0.5) == BFloat16(1)/BFloat16(-2)
    @test BFloat16(1) == BFloat16(0.5)/BFloat16(0.5)
    @test BFloat16(1.5) == BFloat16(3)/BFloat16(2)
    @test BFloat16(-12.5) == BFloat16(-25)/BFloat16(2)
    @test InfB16 == BFloat16(1)/BFloat16(0)
    @test -InfB16 == BFloat16(-1)/BFloat16(0)
end

@testset "Power" begin
    @test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
    @test BFloat16(4) ^ BFloat16(-2) == BFloat16(1/16)
end

@testset "Sqrt" begin
    @test sqrt(BFloat16(4f0)) == BFloat16(2f0)
    @test sqrt(BFloat16(1f0)) == BFloat16(1f0)
    @test sqrt(BFloat16(0.25)) == BFloat16(0.5)
end
