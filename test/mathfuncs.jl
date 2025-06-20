phi = BFloat16(0.0f0 + Base.Math.MathConstants.golden)
invphi = BFloat16(1.0f0 / Base.Math.MathConstants.golden)

for F in  (:abs, :abs2, :sqrt, :cbrt,
          :exp, :exp2, :exp10, :expm1,
          :log, :log2, :log10, :log1p,
          :sin, :cos, :tan, :csc, :sec, :cot,
          :asin, :acos, :atan, :acot,
          :sinh, :cosh, :tanh, :csch, :sech, :coth,
          :asinh, :atanh, :acsch, :asech)
    @eval begin
        @test $F(invphi) == BFloat16($F(Float32(invphi)))
    end
end

@test fma(phi, invphi, invphi) == BFloat16(fma(Float32(phi), Float32(invphi), Float32(invphi)))

for F in (:asec, :acsc, :cosh, :acosh, :acoth)
    @eval begin
        @test $F(phi) == BFloat16($F(Float32(phi)))
    end
end


x,y = rand(Float32, 2)
@test widemul(BFloat16(x), BFloat16(y)) isa Float32
