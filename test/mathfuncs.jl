half = BFloat16(0.5f0)
whole = BFloat16(1.0f0)
two = BFloat16(2.0f0)

phi = BFloat16(0.0f0 + Base.Math.MathConstants.golden)
invphi = BFloat16(1.0f0 / Base.Math.MathConstants.golden)
phi3 = phi * phi * phi
invphi3 = invphi * invphi * invphi

for F in (:abs, :sqrt, :exp, :log, :log2, :log10, :sin, :cos, :tan,
          :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :atanh)
  @eval begin
    @test $F(invphi) == BFloat16($F(Float32(invphi)))
  end
end

@test cosh(phi) == BFloat16(cosh(Float32(phi)))
