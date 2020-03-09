module BFloat16s

    export BFloat16, BFloat16sr,
            BFloat16_stochastic_round,
            NaNB16, InfB16,
            LowPrecArray

    include("bfloat16.jl")
    include("lowprecarrays.jl")

end
