## BFloat16s
----

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/JuliaMath/BFloat16s.jl/stable/)  [![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/JuliaMath/BFloat16s.jl/dev/) [![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT) 

----

This package defines the [BFloat16 data type](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).

Hardware implementation of this datatype is available in Google's
[Cloud TPUs](https://en.wikipedia.org/wiki/Tensor_processing_unit) as well as
in a growing number of CPUs, GPUs, and more specialized processors. See the
[wikipedia entry](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
for more information.

This package is suitable to evaluate whether using BFloat16 would cause
precision problems for any particular algorithm, even without access to supporting
hardware. Note that this package is designed for functionality, not performance,
so this package should be used for precision experiments only, not performance
experiments.

# Usage

This package exports the BFloat16 data type. This datatype should behave
just like any builtin floating point type (e.g. you can construct it from
other floating point types - e.g. `BFloat16(1.0)`). In addition, this package
provides the `LowPrecArray` type. This array is supposed to emulate the kind
of matmul operation that TPUs do well (BFloat16 multiply with Float32
accumulate). Broadcasts and scalar operations are peformed in Float32 (as
they would be on a TPU) while matrix multiplies are performed in BFloat16 with
Float32 accumulates, e.g.

```
julia> A = LowPrecArray(rand(Float32, 5, 5))
5×5 LowPrecArray{2,Array{Float32,2}}:
 0.252818  0.619702   0.553199  0.75225   0.30819
 0.166347  0.976339   0.399945  0.589101  0.526253
 0.350232  0.0447034  0.490874  0.525144  0.841436
 0.903734  0.879541   0.706704  0.304369  0.951702
 0.308417  0.645731   0.65906   0.636451  0.765263

julia> A^2
5×5 LowPrecArray{2,Array{Float32,2}}:
 1.13603   1.64932  1.39712  1.27283  1.82597
 1.03891   1.93298  1.44455  1.42625  1.86842
 0.998384  1.28403  1.37666  1.24076  1.68507
 1.18951   2.33245  2.04367  2.26849  2.35588
 1.22636   1.90367  1.70848  1.63986  2.1826

julia> A.storage^2
5×5 Array{Float32,2}:
 1.13564  1.64708  1.39399  1.27087  1.82128
 1.03924  1.93216  1.44198  1.42456  1.86497
 1.00201  1.28786  1.37826  1.24295  1.6882
 1.19089  2.33262  2.04094  2.26745  2.354
 1.22742  1.90498  1.70653  1.63928  2.18076

julia> Float64.(A.storage)^2
5×5 Array{Float64,2}:
 1.13564  1.64708  1.39399  1.27087  1.82128
 1.03924  1.93216  1.44198  1.42456  1.86497
 1.00201  1.28786  1.37826  1.24295  1.6882
 1.19089  2.33262  2.04094  2.26745  2.354
 1.22742  1.90498  1.70653  1.63928  2.18076
```

Note that the low precision result differs from (is less precise than) the
result computed in Float32 arithmetic (which matches the result in Float64
precision).

