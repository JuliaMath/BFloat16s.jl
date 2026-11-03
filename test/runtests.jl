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

    # Int is only 32 bits on i686; exercise the 64-bit conversions explicitly.
    for T in (Int64, UInt64)
        values = T[0, 1, 255, 256, 257, 258, 259, typemax(T) - 1, typemax(T)]
        # Use Float32-exact inputs around each tie so the reference does not
        # double-round on targets that convert integers directly to bfloat.
        for shift in (0, 16, 32, 54)
            midpoint = T(257) << shift
            step = T(1) << max(0, shift - 15) # Float32 spacing at the midpoint
            append!(values, (midpoint - step, midpoint, midpoint + step))
        end
        if T === Int64
            append!(values, -values)
            append!(values, (typemin(T), typemin(T) + 1))
        end
        for x in values
            @test BFloat16(x) === BFloat16(Float32(x))
        end
    end
    @test reinterpret(UInt16, BFloat16(typemin(Int64))) == 0xdf00
    @test reinterpret(UInt16, BFloat16(typemax(Int64))) == 0x5f00
    @test reinterpret(UInt16, BFloat16(typemax(UInt64))) == 0x5f80
    @test signbit(BFloat16(Int64(-1)))
    @test !signbit(BFloat16(Int64(0)))
end

@testset "correct rounding" begin
    # every finite, non-negative BFloat16, as a Float64
    finite = [Float64(reinterpret(Float32, UInt32(u) << 16)) for u in 0x0000:0x7f7f]
    # round to the nearest BFloat16, ties to even, by comparing against the midpoints
    # between consecutive BFloat16s, which Float64 represents exactly
    function reference(x::Real)
        isnan(x) && return BFloat16s.NaNB16
        a = x isa Integer ? abs(widen(x)) : abs(x)
        i = searchsortedlast(finite, a; lt=<)
        u = UInt16(i - 1)
        if i == length(finite)
            # overflow, at the midpoint between floatmax and the next power of two
            a >= finite[end] + 2.0^119 && (u = 0x7f80)
        else
            mid = (finite[i] + finite[i+1]) / 2
            (a > mid || (a == mid && isodd(u))) && (u += 0x0001)
        end
        return reinterpret(BFloat16, signbit(x) ? u | 0x8000 : u)
    end
    # the conversions, on the host's code path and on the software one. The sign of a NaN
    # result is unspecified: native conversions keep it on AArch64 but not on x86.
    same(y, r) = isnan(r) ? isnan(y) : y === r
    check(x) = same(BFloat16(x), reference(x)) &&
               (x isa Integer ? BFloat16s.bfloat16_from_integer(x) :
                x isa Float64 ? BFloat16s.bfloat16_from_float64(x) :
                BFloat16(x)) === reference(x)

    # values on both sides of a midpoint, closer than Float32 can resolve
    rng = MersenneTwister(0)
    mids = [(finite[i] + finite[i+1]) / 2 for i in rand(rng, 1:length(finite)-1, 1000)]
    floats = [mids; prevfloat.(mids); nextfloat.(mids); randn(rng, 1000)]
    append!(floats, [0.0, nextfloat(0.0), floatmin(Float64), 2.0^-134, 2.0^-133, 2.0^-126,
                     prevfloat(2.0^-126), Inf, NaN, floatmax(Float64),
                     finite[end], finite[end] + 2.0^119, prevfloat(finite[end] + 2.0^119)])
    append!(floats, -floats)
    @test all(check, floats)
    @test BFloat16(1 + 2.0^-8 + 2.0^-30) == BFloat16(1.0078125)
    @test BFloat16(-(1 + 2.0^-8 + 2.0^-30)) == BFloat16(-1.0078125)
    @test signbit(BFloat16(-0.0)) && signbit(BFloat16(-nextfloat(0.0)))
    @test isnan(BFloat16(-NaN))

    for T in (Int32, UInt32, Int64, UInt64, Int128, UInt128)
        # just above a midpoint, beyond Float32's 24 bits of precision
        ints = T[(T(257) << k) + one(T) for k in 16:(8sizeof(T) - 10)]
        append!(ints, rand(rng, T, 1000))
        append!(ints, [typemax(T), typemax(T) - one(T), typemin(T)])
        T <: Signed && append!(ints, -ints[1:end-1])
        @test all(check, ints)
    end
    @test BFloat16(Int64(2)^24 + Int64(2)^16 + 1) == BFloat16(16908288)
    @test BFloat16(typemax(UInt128)) == BFloat16s.InfB16
    @test BFloat16(big(2)^24 + big(2)^16 + 1) == BFloat16(16908288)
    @test BFloat16(big(2)^200) == BFloat16s.InfB16
    @test BFloat16(-big(2)^200) == -BFloat16s.InfB16
    @test BFloat16(true) == one(BFloat16) && BFloat16(false) == zero(BFloat16)

    setprecision(BigFloat, 200) do
        bigs = [big(1) + big(2.0)^-8 + big(2.0)^-60, big(2.0)^-140 * (1 + big(2.0)^-100),
                big(2.0)^-135 + big(2.0)^-190, big(floatmax(Float64)) * 4,
                big(2.0)^-2000, big(0.0)]
        append!(bigs, -bigs)
        @test all(x -> BFloat16(x) === reference(x), bigs)
    end
    setprecision(BigFloat, 12) do
        bigs = [big(1) + big(2.0)^-8 + big(2.0)^-11, big(1) + big(2.0)^-8]
        @test all(x -> BFloat16(x) === reference(x), bigs)
    end
    @test BFloat16(big"1.0039062500000000000001") == BFloat16(1.0078125)
    @test isnan(BFloat16(big(NaN)))
    @test BFloat16(big(Inf)) == BFloat16s.InfB16
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

@testset "control flow" begin
    # On i686, the 64-bit atomic expands to a loop. Native bfloat values carried
    # across its basic blocks can lose their sign, even without coverage.
    Base.@noinline function sign_after_atomic(x::BFloat16, counter::Threads.Atomic{Int64})
        Threads.atomic_add!(counter, Int64(1))
        return signbit(x)
    end
    counter = Threads.Atomic{Int64}(0)
    for bits in UInt16[0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0xffc0]
        @test sign_after_atomic(reinterpret(BFloat16, bits), counter) == (bits >> 15 != 0)
    end
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

    @test nextfloat(BFloat16(1.0), UInt(5)) == nextfloat(BFloat16(1.0), 5)
    @test prevfloat(BFloat16(1.0), UInt(5)) == prevfloat(BFloat16(1.0), 5)
    @test nextfloat(BFloat16(0.0), typemax(UInt64)) == Inf
    @test prevfloat(BFloat16(0.0), typemax(UInt64)) == -Inf
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

@testset "i/o" begin
    @test reinterpret(UInt16, BFloat16(1/3)) == 0x3eab
    @test reinterpret(UInt16, bswap(BFloat16(1/3))) == 0xab3e

    io = IOBuffer()

    write(io, htol(BFloat16(3.14159)))
    @test take!(io) == UInt8[0x49, 0x40]

    write(io, hton(BFloat16(3.14159)))
    @test take!(io) == UInt8[0x40, 0x49]

    @test htol(read(IOBuffer(UInt8[0x49, 0x40]), BFloat16)) == BFloat16(3.14159)
    @test hton(read(IOBuffer(UInt8[0x40, 0x49]), BFloat16)) == BFloat16(3.14159)
end

include("structure.jl")
include("mathfuncs.jl")
include("lowprecarrays.jl")

end # @testset "BFloat16s"
