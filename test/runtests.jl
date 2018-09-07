using BFloat16s
using Test

half = 0.5f0
two  = 2.0f0

bfhalf = BFloat16(0.5)
bf1 = BFloat16(1.0)
bf2 = BFloat16(2.0)
bf3 = BFloat16(3.0)
bf4 = BFloat16(4.0)
bfneg1 = BFloat16(-1.0)

@test bfhalf == bfhalf
@test bf1 != bf2
@test bf1 > bfneg1
@test bf1 >= bfneg1
@test bf2 > bfneg1
@test bf2 >= bf1

@test bf1 + bf2 == bf3
@test bf2 - bf1 == bf1
@test bf2 * bf2 == bf4
@test bf4 / bf2 == bf2

for F in (:sqrt, :cbrt, :exp, :expm1, :log, :log1p, :log10, :log2,
          :sin, :cos, :tan, :csc, :sec, :cot,
          :asin, :acos, :atan, :acot,
          :sinh, :cosh, :tanh, :csch, :sech, :coth,
          :asinh, :atanh, :acsch, :asech)
  @eval begin
    @test isapprox($F(bfhalf), $F(half))
  end
end

for F in (:acsc, :asec,
          :acosh,:acoth)
  @eval begin
    @test isapprox($F(bf2), $F(two))
  end
end

    
