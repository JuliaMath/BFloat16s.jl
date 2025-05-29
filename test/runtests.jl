using Test, BFloat16s, Printf, Random

@info "Testing BFloat16s" BFloat16s.llvm_storage BFloat16s.llvm_arithmetic

@testset "BFloat16s" begin

@testset "basics" begin
    @test Base.exponent_bits(BFloat16) == 8
    @test Base.significand_bits(BFloat16) == 7
    @test precision(BFloat16) == 8
    @test Base.uinttype(BFloat16) == UInt16

    @test typemin(BFloat16) == -BFloat16s.InfB16
    @test typemax(BFloat16) == BFloat16s.InfB16
end

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
    @test BFloat16(NaN) != BFloat16(1.0)
    @test !(BFloat16(1.0) == BFloat16(NaN))
    @test iszero(BFloat16(0)) == true
    @test iszero(BFloat16(3.45)) == false
end

@testset "conversions" begin
    @test Float32(BFloat16(10)) == 1f1
    @test Float64(BFloat16(10)) == 10.0
    @test Int32(BFloat16(10)) == Int32(10)
    @test UInt32(BFloat16(10)) == Int32(10)
    @test Int64(BFloat16(10)) == Int64(10)
    @test UInt64(BFloat16(10)) == Int64(10)
    @test BFloat16(BigFloat(1)) == BFloat16(1)
    @test BigFloat(BFloat16(1)) == BigFloat(1)
    @test Float16(BFloat16(3.140625)) == Float16(π)
    @test BFloat16(Float16(π)) == BFloat16(3.140625)
    @test BFloat16(pi) == BFloat16(3.14159)
    @test all(R -> R<:BFloat16, Base.return_types(BFloat16))

    @test promote(BFloat16(4.5), Float64(5.0)) == (Float64(4.5), Float64(5.0))
    @test promote(BFloat16(4.5), Float32(5.0)) == (Float32(4.5), Float32(5.0))

    @test_throws InexactError Int16(typemax(BFloat16))
    @test_throws InexactError UInt16(typemax(BFloat16))
end

@testset "trunc" begin
    f_val = 5 .+ rand(100)
    bf_val = BFloat16.(f_val)

    @testset "$Ts, $(unsigned(Ts))" for Ts in (Int8, Int16, Int32, Int64, Int128)
        @test trunc.(Ts, bf_val) == trunc.(Ts, bf_val)
        @test trunc.(Ts, -bf_val) == trunc.(Ts, -bf_val)

        Tu = unsigned(Ts)
        @test trunc.(Tu, bf_val) == trunc.(Tu, bf_val)
    end

    @test trunc(BFloat16, Float32(π)) == BFloat16(3.140625)

    # InexactError
    @test_throws InexactError trunc(Int16, typemax(BFloat16))
    @test_throws InexactError trunc(UInt16, typemax(BFloat16))
end

@testset "abi" begin
    f() = BFloat16(1)
    @test f() == BFloat16(1)

    g(x) = x+BFloat16(1)
    @test g(BFloat16(2)) == BFloat16(3)
end

@testset "functions" begin
    @test abs(BFloat16(-10)) == BFloat16(10)
    @test BFloat16(2) ^ BFloat16(4) == BFloat16(16)
    @test eps(BFloat16) == BFloat16(0.0078125)
    @test sqrt(BFloat16(4f0)) == BFloat16(2f0)
    @test rem(BFloat16(3.14), Int) == 3
    @test round(BFloat16(10.4), RoundToZero) == BFloat16(10.0)
    @test round(BFloat16(10.4), RoundUp) == BFloat16(11.0)
    @test round(BFloat16(10.6), RoundDown) == BFloat16(10.0)
    @test round(BFloat16(3.2), RoundNearest) == BFloat16(3.0)
    @test round(BFloat16(4.8), RoundNearest) == BFloat16(5.0)
end

@testset "arithmetic" begin
    @test BFloat16(0.2) * BFloat16(5.0) == BFloat16(1.0)
    @test BFloat16(1.0) / BFloat16(5.0) == BFloat16(0.2)
    @test inv(BFloat16(5.0)) == BFloat16(0.2)
    @test zero(BFloat16) == BFloat16(0.0f0)
    @test one(BFloat16) == BFloat16(1.0)
    @test BFloat16(2.0) ^ -2 == BFloat16(0.25)
end

@testset "printf" begin
    for (fmt, val) in (("%7.2f",  "   1.23"),
                       ("%-7.2f",    "1.23   "),
                       ("%07.2f", "0001.23"),
                       ("%.0f",      "1"),
                       ("%#.0f",     "1."),
                       ("%.3e",      "1.234e+00"),
                       ("%.3E",      "1.234E+00"),
                       ("%.2a",    "0x1.3cp+0"),
                       ("%.2A",    "0X1.3CP+0")),
            num in (BFloat16(1.234),)
        @eval @test @sprintf($fmt, $num) == $val
    end
    @test (@sprintf "%f" BFloat16(Inf)) == "Inf"
    @test (@sprintf "%f" BFloat16(NaN)) == "NaN"
    @test (@sprintf "%.0e" BFloat16(3e2)) == "3e+02"
    @test (@sprintf "%#.0e" BFloat16(3e2)) == "3.e+02"

    for (fmt, val) in (("%10.5g",  "     123.5"),
                       ("%+10.5g", "    +123.5"),
                       ("% 10.5g", "     123.5"),
                       ("%#10.5g",  "    123.50"),
                       ("%-10.5g",      "123.5     "),
                       ("%-+10.5g",    "+123.5    "),
                       ("%010.5g", "00000123.5")),
        num in (BFloat16(123.5),)
        @eval @test @sprintf($fmt, $num) == $val
    end
    @test( @sprintf( "%10.5g", BFloat16(-123.5) ) == "    -123.5")
    @test( @sprintf( "%010.5g", BFloat16(-123.5) ) == "-0000123.5")
    @test (@sprintf "%a" BFloat16(1.5)) == "0x1.8p+0"
end

@testset "show" begin
    @test repr(BFloat16(Inf)) == "InfB16"
    @test repr(BFloat16(-Inf)) == "-InfB16"
    @test repr(BFloat16(NaN)) == "NaNB16"
    @test repr(BFloat16(2)) == "BFloat16(2.0)"
end

@testset "parse" for _parse in (parse, tryparse)
    @test _parse(BFloat16, "Inf") === BFloat16(Inf)
    @test _parse(BFloat16, "-Inf") === BFloat16(-Inf)
    nan16 = _parse(BFloat16, "NaN")
    @test isnan(nan16)
    @test isa(nan16, BFloat16)
    @test _parse(BFloat16, "2") === BFloat16(2)
    @test _parse(BFloat16, "1.3") === BFloat16(1.3)
    @test _parse(BFloat16, "+234.6") === BFloat16(234.6)
    @test _parse(BFloat16, "    +234.7") === BFloat16(234.7)
    @test _parse(BFloat16, "    -234.8") === BFloat16(-234.8)
    @test _parse(BFloat16, "    -234.90") === BFloat16(-234.9)
    @test _parse(BFloat16, "    235.10 ") === BFloat16(235.1)
    @test _parse(BFloat16, "000235.20 ") === BFloat16(235.2)
    @test _parse(BFloat16, "4e+03") === BFloat16(4e3)
    @test _parse(BFloat16, "5.e+04") === BFloat16(5e4)
    @test _parse(BFloat16, "0x1.3cp+0") === BFloat16(1.234375)
    @test _parse(BFloat16, "0X1.3CP+0") === BFloat16(1.234375)
end

@testset "not parseable" for str in ("635.3X", "X635.4", "ABCDE", "1e0e0")
    @test tryparse(BFloat16, str) === nothing
    @test_throws ArgumentError parse(BFloat16, str)
end

@testset "random" begin
    x = Array{BFloat16}(undef, 10)
    y = Array{BFloat16}(undef, 10)
    rand!(x)
    rand!(y)
    @test x !== y

    randn!(x)
    randn!(y)
    @test x !== y

    randexp!(x)
    randexp!(y)
    @test x !== y

    x = rand(BFloat16, 10)
    y = rand(BFloat16, 10)
    @test x !== y

    x = randn(BFloat16, 10)
    y = randn(BFloat16, 10)
    @test x !== y

    x = randexp(BFloat16, 10)
    y = randexp(BFloat16, 10)
    @test x !== y
end

@testset "round" begin
    @test round(Int, BFloat16(3.4)) == 3
end

@testset "Next/prevfloat" begin
    for x in (one(BFloat16),
                -one(BFloat16),
                zero(BFloat16))
        @test x == nextfloat(prevfloat(x))
        @test x == prevfloat(nextfloat(x))

        @test x < nextfloat(x)
        @test x > prevfloat(x)

        @test nextfloat(x, typemax(Int)) == typemax(BFloat16)
        @test prevfloat(x, typemax(Int)) == typemin(BFloat16)
    end

    @test isnan(nextfloat(BFloat16s.NaNB16))
    @test isinf(nextfloat(BFloat16s.InfB16))

    @test isnan(prevfloat(BFloat16s.NaNB16))
end

@testset "Decompose BFloat16" begin
    for x in randn(100)
        bf16 = BFloat16(x)
        s,e,d = Base.decompose(bf16)
        @test BFloat16(s*2.0^e/d) == bf16
    end
end


@testset "Next/prevfloat(x,::Integer)" begin
    x = one(BFloat16)
    @test x == prevfloat(nextfloat(x,100),100)
    @test x == nextfloat(prevfloat(x,100),100)

    x = -one(BFloat16)
    @test x == prevfloat(nextfloat(x,100),100)
    @test x == nextfloat(prevfloat(x,100),100)

    x = one(BFloat16)
    @test nextfloat(x,5) == prevfloat(x,-5)
    @test prevfloat(x,-5) == nextfloat(x,5)

    @test isinf(nextfloat(floatmax(BFloat16),5))
    @test prevfloat(floatmin(BFloat16),2^8) < 0
    @test nextfloat(-floatmin(BFloat16),2^8) > 0
end

@testset "maxintfloat" begin
    a = maxintfloat(BFloat16)
    @test a+1-1 == a-1    # the first +1 cannot be represented
    @test a-1+1 == a      # but -1 can
end

@testset "rand sampling" begin
    Random.seed!(123)
    mi, ma = extrema(rand(BFloat16, 1_000_000))

    # zero should be the lowest BFloat16 sampled
    @test mi === zero(BFloat16)

    # prevfloat(one(BFloat16)) should be maximum
    @test ma === prevfloat(one(BFloat16), 1)
end

include("structure.jl")
include("mathfuncs.jl")
include("lowprecarrays.jl")

end # @testset "BFloat16s"
