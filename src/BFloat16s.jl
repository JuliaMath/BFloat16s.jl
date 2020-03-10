module BFloat16s

    export BFloat16, BFloat16sr,
            BFloat16_stochastic_round,
            NaNB16, InfB16,
            LowPrecArray

    using RandomNumbers.Xorshifts

    RNG = Xoroshiro128Plus()

    include("bfloat16.jl")
    include("lowprecarrays.jl")

end
