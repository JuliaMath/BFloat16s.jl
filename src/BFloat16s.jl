module BFloat16s

    export BFloat16, BFloat16sr,
            NaNB16, InfB16,
            LowPrecArray

    import Base: isfinite, isnan, precision, iszero,
        sign_mask, exponent_mask, exponent_one, exponent_half,
        significand_mask,
        +, -, *, /, ^

    include("bfloat16.jl")
    include("lowprecarrays.jl")

end
