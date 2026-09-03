using BFloat16s

values = Float32.(1:64)
ints = BFloat16.(1:64)
floats = BFloat16.(values)
sums = floats + floats

@assert Float32.(ints) == values
@assert Float32.(floats) == values
@assert Float32.(sums) == 2 .* values
