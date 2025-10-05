import Base: isfinite, isnan, precision, iszero, eps,
    typemin, typemax, floatmin, floatmax,
    sign_mask, exponent_mask, significand_mask,
    exponent_bits, significand_bits, exponent_bias,
    exponent_one, exponent_half, leading_zeros,
    signbit, exponent, significand, frexp, ldexp,
    round, Int16, Int32, Int64,
    +, -, *, /, ^, ==, <, <=, inv,
    abs, abs2, uabs, sqrt, cbrt,
    exp, exp2, exp10, expm1,
    log, log2, log10, log1p,
    sin, cos, tan, csc, sec, cot,
    asin, acos, atan, acsc, asec, acot,
    sinh, cosh, tanh, csch, sech, coth,
    asinh, acosh, atanh, acsch, asech, acoth,
    bitstring, isinteger

import Printf

# LLVM 11 added support for BFloat16 in the IR; Julia 1.11 added support for generating
# code that uses the `bfloat` IR type, together with the necessary runtime functions.
# However, not all LLVM targets support `bfloat`. If the target can store/load BFloat16s
# (and supports synthesizing constants) we can use the `bfloat` IR type, otherwise we fall
# back to defining a primitive type that will be represented as an `i16`. If, in addition,
# the target supports BFloat16 arithmetic, we can use LLVM instructions.
# - x86: storage and arithmetic support in LLVM 15
# - aarch64: storage support in LLVM 17
const llvm_storage = if isdefined(Core, :BFloat16)
    if Sys.ARCH in [:x86_64, :i686] && Base.libllvm_version >= v"15"
        true
    elseif Sys.ARCH == :aarch64 && Base.libllvm_version >= v"17"
        true
    else
        false
    end
else
    false
end
if llvm_storage
    import Core: BFloat16
end
const llvm_arithmetic = if llvm_storage
    if Sys.ARCH in [:x86_64, :i686] && Base.libllvm_version >= v"15"
        true
    elseif Sys.ARCH == :aarch64 && Base.libllvm_version >= v"19"
        true
    else
        false
    end
else
    primitive type BFloat16 <: AbstractFloat 16 end
    false
end

Base.reinterpret(::Type{Unsigned}, x::BFloat16) = reinterpret(UInt16, x)
Base.reinterpret(::Type{Signed}, x::BFloat16) = reinterpret(Int16, x)

# Floating point property queries
for f in (:sign_mask, :exponent_mask, :exponent_one,
          :exponent_half, :significand_mask)
    @eval $(f)(::Type{BFloat16}) = UInt16($(f)(Float32) >> 16)
end
Base.exponent_bias(::Type{BFloat16}) = 127
Base.exponent_bits(::Type{BFloat16}) = 8
Base.significand_bits(::Type{BFloat16}) = 7
Base.signbit(x::BFloat16) = (reinterpret(Unsigned, x) & 0x8000) !== 0x0000

function Base.issubnormal(x::BFloat16)
    y = reinterpret(Unsigned, x)
    return (y & exponent_mask(BFloat16) == 0) & (y & significand_mask(BFloat16) != 0)
end

function Base.significand(x::BFloat16)
    xu = reinterpret(Unsigned, x)
    xs = xu & ~sign_mask(BFloat16)
    xs >= exponent_mask(BFloat16) && return x # NaN or Inf
    if xs <= (~exponent_mask(BFloat16) & ~sign_mask(BFloat16)) # x is subnormal
        xs == 0 && return x # +-0
        m = unsigned(leading_zeros(xs) - exponent_bits(BFloat16))
        xs <<= m
        xu = xs | (xu & sign_mask(BFloat16))
    end
    xu = (xu & ~exponent_mask(BFloat16)) | exponent_one(BFloat16)
    return reinterpret(BFloat16, xu)
end

Base.exponent(x::BFloat16) =
    ((reinterpret(Unsigned, x) & Base.exponent_mask(BFloat16)) >> 7) - Base.exponent_bias(BFloat16)

function Base.decompose(x::BFloat16)::NTuple{3,Int}
    isnan(x) && return 0, 0, 0
    isinf(x) && return ifelse(x < 0, -1, 1), 0, 0
    n = reinterpret(UInt16, x)
    s = (n & 0x007f) % Int16
    e = ((n & 0x7f80) >> 7) % Int
    s |= Int16(e != 0) << 7
    d = ifelse(signbit(x), -1, 1)
    s, e - 134 + (e == 0), d
end

function Base.frexp(x::BFloat16)
   xp = exponent(x) + 1
   fr = significand(x) * BFloat16(0.5)
   (fr, xp)
end

function Base.ldexp(fr::BFloat16, xp::Integer)
   fr * BFloat16(2)^(xp)
end

function Base.rem(x::BFloat16, ::Type{T}) where {T<:Integer}
    T(trunc(x))
end

iszero(x::BFloat16) = reinterpret(Unsigned, x) & ~sign_mask(BFloat16) == 0x0000
isfinite(x::BFloat16) = (reinterpret(Unsigned,x) & exponent_mask(BFloat16)) != exponent_mask(BFloat16)
isnan(x::BFloat16) = (reinterpret(Unsigned,x) & ~sign_mask(BFloat16)) > exponent_mask(BFloat16)
precision(::Type{BFloat16}) = 8
eps(::Type{BFloat16}) = Base.bitcast(BFloat16, 0x3c00)

## Rounding ##
if llvm_arithmetic
    round(x::BFloat16, ::RoundingMode{:ToZero})  = Base.trunc_llvm(x)
    round(x::BFloat16, ::RoundingMode{:Down})    = Base.floor_llvm(x)
    round(x::BFloat16, ::RoundingMode{:Up})      = Base.ceil_llvm(x)
    round(x::BFloat16, ::RoundingMode{:Nearest}) = Base.rint_llvm(x)
else
    round(x::BFloat16, r::RoundingMode{:ToZero}) = BFloat16(trunc(Float32(x)))
    round(x::BFloat16, r::RoundingMode{:Down}) = BFloat16(floor(Float32(x)))
    round(x::BFloat16, r::RoundingMode{:Up}) = BFloat16(ceil(Float32(x)))
    round(x::BFloat16, r::RoundingMode{:Nearest}) = BFloat16(round(Float32(x)))
end
# round(::Type{Signed},   x::BFloat16, r::RoundingMode) = round(Int, x, r)
# round(::Type{Unsigned}, x::BFloat16, r::RoundingMode) = round(UInt, x, r)
# round(::Type{Integer},  x::BFloat16, r::RoundingMode) = round(Int, x, r)

Base.trunc(bf::BFloat16) = signbit(bf) ? ceil(bf) : floor(bf)

## floating point traits ##
"""
    InfB16
Positive infinity of type [`BFloat16`](@ref).
"""
const InfB16 = reinterpret(BFloat16, 0x7f80)

"""
    NaNB16
A not-a-number value of type [`BFloat16`](@ref).
"""
const NaNB16 = reinterpret(BFloat16, 0x7fc0)

# More floating point property queries
typemin(::Type{BFloat16}) = -InfB16
typemax(::Type{BFloat16}) = InfB16
floatmax(::Type{BFloat16}) = reinterpret(BFloat16, 0x7f7f)
floatmin(::Type{BFloat16}) = reinterpret(BFloat16, 0x0080)
Base.maxintfloat(::Type{BFloat16}) = reinterpret(BFloat16,0x4380) # = BFloat16(256)

# Truncation from Float32
Base.uinttype(::Type{BFloat16}) = UInt16
Base.trunc(::Type{BFloat16}, x::Float32) = reinterpret(BFloat16,
        (reinterpret(UInt32, x) >> 16) % UInt16
    )

if llvm_arithmetic
    BFloat16(x::Float32) = Base.fptrunc(BFloat16, x)
    BFloat16(x::Float64) = Base.fptrunc(BFloat16, x)

    # XXX: can LLVM do this natively?
    BFloat16(x::Float16) = BFloat16(Float32(x))
else
    # Conversion from Float32
    function BFloat16(x::Float32)
        isnan(x) && return NaNB16
        # Round to nearest even (matches TensorFlow and our convention for
        # rounding to lower precision floating point types).
        h = reinterpret(UInt32, x)
        h += 0x7fff + ((h >> 16) & 1)
        return reinterpret(BFloat16, (h >> 16) % UInt16)
    end

    # Conversion from Float64
    function BFloat16(x::Float64)
        BFloat16(Float32(x))
    end

    # Conversion from Float16
    function BFloat16(x::Float16)
        BFloat16(Float32(x))
    end
end

# Conversion from Integer
if llvm_arithmetic
    for st in (Int8, Int16, Int32, Int64)
        @eval begin
            BFloat16(x::($st)) = Base.sitofp(BFloat16, x)
        end
    end
    for ut in (Bool, UInt8, UInt16, UInt32, UInt64)
        @eval begin
            BFloat16(x::($ut)) = Base.uitofp(BFloat16, x)
        end
    end
else
    BFloat16(x::Integer) = convert(BFloat16, convert(Float32, x)::Float32)
end
# TODO: optimize
BFloat16(x::UInt128) = convert(BFloat16, Float64(x))
BFloat16(x::Int128)  = convert(BFloat16, Float64(x))

# Conversion to Float16
function Base.Float16(x::BFloat16)
    Float16(Float32(x))
end

if llvm_arithmetic
    Base.Float32(x::BFloat16) = Base.fpext(Float32, x)
    Base.Float64(x::BFloat16) = Base.fpext(Float64, x)
else
    # Expansion to Float32
    function Base.Float32(x::BFloat16)
        reinterpret(Float32, UInt32(reinterpret(Unsigned, x)) << 16)
    end

    # Expansion to Float64
    function Base.Float64(x::BFloat16)
        Float64(Float32(x))
    end
end

# BigFloat conversion
BFloat16(x::BigFloat) = BFloat16(Float32(x))
Base.BigFloat(x::BFloat16) = BigFloat(Float32(x))

# Basic arithmetic
if llvm_arithmetic
    +(x::T, y::T) where {T<:BFloat16} = Base.add_float(x, y)
    -(x::T, y::T) where {T<:BFloat16} = Base.sub_float(x, y)
    *(x::T, y::T) where {T<:BFloat16} = Base.mul_float(x, y)
    /(x::T, y::T) where {T<:BFloat16} = Base.div_float(x, y)
    -(x::BFloat16) = Base.neg_float(x)
    ^(x::BFloat16, y::BFloat16) = BFloat16(Float32(x)^Float32(y))
else
    for f in (:+, :-, :*, :/, :^)
        @eval ($f)(x::BFloat16, y::BFloat16) = BFloat16($(f)(Float32(x), Float32(y)))
    end
    -(x::BFloat16) = reinterpret(BFloat16, reinterpret(Unsigned, x) ⊻ sign_mask(BFloat16))
end
^(x::BFloat16, y::Integer) = BFloat16(Float32(x)^y)

const ZeroBFloat16 = BFloat16(0.0f0)
const OneBFloat16 = BFloat16(1.0f0)
Base.zero(::Type{BFloat16}) = ZeroBFloat16
Base.one(::Type{BFloat16}) = OneBFloat16

inv(x::BFloat16) = one(BFloat16) / x

# Floating point comparison
function ==(x::BFloat16, y::BFloat16)
    ix = reinterpret(Unsigned, x)
    iy = reinterpret(Unsigned, y)
    # NaNs (isnan(x) || isnan(y))
    if (ix|iy)&~sign_mask(BFloat16) > exponent_mask(BFloat16)
        return false
    end
    # Signed zeros
    if (ix|iy)&~sign_mask(BFloat16) == 0
        return true
    end
    return ix == iy
end

<(a::BFloat16, b::BFloat16) = Float32(a) < Float32(b)
@static if VERSION < v"1.8"
    <=(a::BFloat16, b::BFloat16) = Float32(a) <= Float32(b)
end

Base.widen(::Type{BFloat16}) = Float32
Base.promote_rule(::Type{Float32}, ::Type{BFloat16}) = Float32
Base.promote_rule(::Type{Float64}, ::Type{BFloat16}) = Float64
for t in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval Base.promote_rule(::Type{BFloat16}, ::Type{$t}) = BFloat16
end

# Wide multiplication
Base.widemul(x::BFloat16, y::BFloat16) = widen(x) * widen(y)

# Truncation to integer types
if llvm_arithmetic
    for Ti in (Int8, Int16, Int32, Int64)
        @eval begin
            Base.unsafe_trunc(::Type{$Ti}, x::BFloat16) = Base.fptosi($Ti, x)
        end
    end
    for Ti in (UInt8, UInt16, UInt32, UInt64)
        @eval begin
            Base.unsafe_trunc(::Type{$Ti}, x::BFloat16) = Base.fptoui($Ti, x)
        end
    end
    Base.unsafe_trunc(::Type{UInt128}, x::BFloat16) = unsafe_trunc(UInt128, Float32(x))
    Base.unsafe_trunc(::Type{Int128}, x::BFloat16) = unsafe_trunc(Int128, Float32(x))
else
    Base.unsafe_trunc(T::Type{<:Integer}, x::BFloat16) = unsafe_trunc(T, Float32(x))
end
for Ti in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    if Ti <: Unsigned || sizeof(Ti) < 2
        # Here `BFloat16(typemin(Ti))-1` is exact, so we can compare the lower-bound
        # directly. `BFloat16(typemax(Ti))+1` is either always exactly representable, or
        # rounded to `Inf` (e.g. when `Ti==UInt128 && BFloat16==Float32`).
        @eval begin
            function Base.trunc(::Type{$Ti}, x::BFloat16)
                if $(BFloat16(typemin(Ti))-one(BFloat16)) < x < $(BFloat16(typemax(Ti))+one(BFloat16))
                    return Base.unsafe_trunc($Ti,x)
                else
                    throw(InexactError(:trunc, $Ti, x))
                end
            end
            function (::Type{$Ti})(x::BFloat16)
                if ($(BFloat16(typemin(Ti))) <= x <= $(BFloat16(typemax(Ti)))) && isinteger(x)
                    return Base.unsafe_trunc($Ti,x)
                else
                    throw(InexactError($(Expr(:quote,Ti.name.name)), $Ti, x))
                end
            end
        end
    else
        # Here `eps(BFloat16(typemin(Ti))) > 1`, so the only value which can be
        # truncated to `BFloat16(typemin(Ti)` is itself. Similarly,
        # `BFloat16(typemax(Ti))` is inexact and will be rounded up. This assumes that
        # `BFloat16(typemin(Ti)) > -Inf`, which is true for these types, but not for
        # `Float16` or larger integer types.
        @eval begin
            function Base.trunc(::Type{$Ti}, x::BFloat16)
                if $(BFloat16(typemin(Ti))) <= x < $(BFloat16(typemax(Ti)))
                    return unsafe_trunc($Ti,x)
                else
                    throw(InexactError(:trunc, $Ti, x))
                end
            end
            function (::Type{$Ti})(x::BFloat16)
                if ($(BFloat16(typemin(Ti))) <= x < $(BFloat16(typemax(Ti)))) && isinteger(x)
                    return unsafe_trunc($Ti,x)
                else
                    throw(InexactError($(Expr(:quote,Ti.name.name)), $Ti, x))
                end
            end
        end
    end
end

# Showing
function Base.show(io::IO, x::BFloat16)
    hastypeinfo = BFloat16 === get(io, :typeinfo, Any)
    if isinf(x)
        print(io, x < 0 ? "-InfB16" : "InfB16")
    elseif isnan(x)
        print(io, "NaNB16")
    else
        hastypeinfo || print(io, "BFloat16(")
        show(IOContext(io, :typeinfo=>Float32), Float32(x))
        hastypeinfo || print(io, ")")
    end
end
Printf.tofloat(x::BFloat16) = Float32(x)

# Parsing
Base.parse(::Type{BFloat16}, s::AbstractString) = BFloat16(parse(Float32, s))
function Base.tryparse(::Type{BFloat16}, s::AbstractString)
    r = tryparse(Float32, s)
    return isnothing(r) ? nothing : BFloat16(r)
end

# Random
import Random: rand, randn, randexp, AbstractRNG, Sampler

"""Sample a BFloat16 from [0,1) by creating a random integer
in 0, ... 255 and then scaling it into [0,1). This samples
from every BFloat16 in [1/2, 1) but only from every other
in [1/4, 1/2), and every forth in [1/8, 1/4), etc. for a
uniform distribution."""
function rand(rng::AbstractRNG, ::Sampler{BFloat16})
    # 0x1p-8 is 2^-8 = eps(BFloat16)/2
    return rand(rng, UInt8) * BFloat16(0x1p-8)
end

randn(rng::AbstractRNG, ::Type{BFloat16}) = convert(BFloat16, randn(rng))
randexp(rng::AbstractRNG, ::Type{BFloat16}) = convert(BFloat16, randexp(rng))

# Bitstring
bitstring(x::BFloat16) = bitstring(reinterpret(Unsigned, x))

# next/prevfloat
function Base.nextfloat(f::BFloat16, d::Integer)
    F = typeof(f)
    fumax = reinterpret(Unsigned, F(Inf))
    U = typeof(fumax)

    isnan(f) && return f
    fi = reinterpret(Signed, f)
    fneg = fi < 0
    fu = unsigned(fi & typemax(fi))

    dneg = d < 0
    da = uabs(d)
    if da > typemax(U)
        fneg = dneg
        fu = fumax
    else
        du = da % U
        if fneg ⊻ dneg
            if du > fu
                fu = min(fumax, du - fu)
                fneg = !fneg
            else
                fu = fu - du
            end
        else
            if fumax - fu < du
                fu = fumax
            else
                fu = fu + du
            end
        end
    end
    if fneg
        fu |= sign_mask(F)
    end
    reinterpret(F, fu)
end

# math functions
for F in (:abs, :abs2, :sqrt, :cbrt,
          :exp, :exp2, :exp10, :expm1,
          :log, :log2, :log10, :log1p,
          :sin, :cos, :tan, :csc, :sec, :cot,
          :asin, :acos, :atan, :acsc, :asec, :acot,
          :sinh, :cosh, :tanh, :csch, :sech, :coth,
          :asinh, :acosh, :atanh, :acsch, :asech, :acoth)
  @eval begin
     Base.$F(x::BFloat16) = BFloat16($F(Float32(x)))
  end
end

# irrationals
BFloat16(x::AbstractIrrational) = BFloat16(Float32(x)::Float32)
