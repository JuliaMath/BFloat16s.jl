module BFloat16s

    export BFloat16, NaNB16, InfB16,
            LowPrecArray

    import Base: isfinite, isnan, precision, iszero,
			sign_mask, exponent_mask, significand_mask,
			significand_bits,
			+, -, *, /, ^,
			nextfloat,prevfloat,one,zero,eps,
			typemin,typemax,floatmin,floatmax,
			==,<=,<,
			Float16,Float32,Float64,
			promote_rule, round,
			Int64,Int32,Int16,Int8,
			UInt64,UInt32,UInt16,UInt8,
			bitstring,show,widen,widemul

    include("bfloat16.jl")
    include("lowprecarrays.jl")

end
