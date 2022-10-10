import Base: isfinite, isnan, precision, iszero, eps,
    typemin, typemax, floatmin, floatmax,
    sign_mask, exponent_mask, significand_mask,
    exponent_bits, significand_bits, exponent_bias,
    exponent_one, exponent_half,
    signbit, exponent, significand, frexp, ldexp,
    round, Int16, Int32, Int64,
    +, -, *, /, ^, ==, <, <=, >=, >, !=, inv,
    abs, abs2, sqrt, cbrt,
    exp, exp2, exp10, expm1,
    log, log2, log10, log1p,
    sin, cos, tan, csc, sec, cot,
    asin, acos, atan, acsc, asec, acot,
    sinh, cosh, tanh, csch, sech, coth,
    asinh, acosh, atanh, acsch, asech, acoth,
    bitstring

primitive type BFloat16 <: AbstractFloat 16 end

# Floating point property queries
for f in (:sign_mask, :exponent_mask, :exponent_one,
            :exponent_half, :significand_mask)
    @eval $(f)(::Type{BFloat16}) = UInt16($(f)(Float32) >> 16)
end

Base.exponent_bias(::Type{BFloat16}) = 127
Base.exponent_bits(::Type{BFloat16}) = 8
Base.significand_bits(::Type{BFloat16}) = 7
Base.signbit(x::BFloat16) = (reinterpret(UInt16, x) & 0x8000) !== 0x0000

function Base.significand(x::BFloat16)
    result = abs_significand(x)
    ifelse(signbit(x), -result, result)
end

@inline function abs_significand(x::BFloat16)
    usig = Base.significand_mask(BFloat16) & reinterpret(UInt16, x)
    isig = Int16(usig)
    1 + isig / BFloat16(2)^7
end

Base.exponent(x::BFloat16) =
    ((reinterpret(UInt16, x) & Base.exponent_mask(BFloat16)) >> 7) - Base.exponent_bias(BFloat16)

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

iszero(x::BFloat16) = reinterpret(UInt16, x) & ~sign_mask(BFloat16) == 0x0000
isfinite(x::BFloat16) = (reinterpret(UInt16,x) & exponent_mask(BFloat16)) != exponent_mask(BFloat16)
isnan(x::BFloat16) = (reinterpret(UInt16,x) & ~sign_mask(BFloat16)) > exponent_mask(BFloat16)
precision(::Type{BFloat16}) = 8
eps(::Type{BFloat16}) = Base.bitcast(BFloat16, 0x3c00)

round(x::BFloat16, r::RoundingMode{:Up}) = BFloat16(ceil(Float32(x)))
round(x::BFloat16, r::RoundingMode{:Down}) = BFloat16(floor(Float32(x)))
round(x::BFloat16, r::RoundingMode{:Nearest}) = BFloat16(round(Float32(x)))

Base.trunc(bf::BFloat16) = signbit(bf) ? ceil(bf) : floor(bf)

Int64(x::BFloat16) = Int64(Float32(x))
Int32(x::BFloat16) = Int32(Float32(x))
Int16(x::BFloat16) = Int16(Float32(x))

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

# Truncation from Float32
Base.uinttype(::Type{BFloat16}) = UInt16
Base.trunc(::Type{BFloat16}, x::Float32) = reinterpret(BFloat16,
        (reinterpret(UInt32, x) >> 16) % UInt16
    )

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

# Conversion from Integer
function BFloat16(x::Integer)
    convert(BFloat16, convert(Float32, x))
end

# Conversion to Float16
function Base.Float16(x::BFloat16)
    Float16(Float32(x))
end

# Expansion to Float32
function Base.Float32(x::BFloat16)
    reinterpret(Float32, UInt32(reinterpret(UInt16, x)) << 16)
end

# Expansion to Float64
function Base.Float64(x::BFloat16)
    Float64(Float32(x))
end

# Truncation to integer types
Base.unsafe_trunc(T::Type{<:Integer}, x::BFloat16) = unsafe_trunc(T, Float32(x))
Base.trunc(::Type{T}, x::BFloat16) where {T<:Integer} = trunc(T, Float32(x))

# Basic arithmetic
for f in (:+, :-, :*, :/, :^)
    @eval ($f)(x::BFloat16, y::BFloat16) = BFloat16($(f)(Float32(x), Float32(y)))
end
-(x::BFloat16) = reinterpret(BFloat16, reinterpret(UInt16, x) ⊻ sign_mask(BFloat16))
^(x::BFloat16, y::Integer) = BFloat16(^(Float32(x), y))

const ZeroBFloat16 = BFloat16(0.0f0)
const OneBFloat16 = BFloat16(1.0f0)
Base.zero(::Type{BFloat16}) = ZeroBFloat16
Base.one(::Type{BFloat16}) = OneBFloat16

inv(x::BFloat16) = one(BFloat16) / x

# Floating point comparison
function ==(x::BFloat16, y::BFloat16)
    ix = reinterpret(UInt16, x)
    iy = reinterpret(UInt16, y)
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

for op in (:<, :<=, :>, :>=, :!=)
    @eval ($op)(a::BFloat16, b::BFloat16) = ($op)(Float32(a), Float32(b))
end

Base.widen(::Type{BFloat16}) = Float32
Base.promote_rule(::Type{Float32}, ::Type{BFloat16}) = Float32
Base.promote_rule(::Type{Float64}, ::Type{BFloat16}) = Float64
for t in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval Base.promote_rule(::Type{BFloat16}, ::Type{$t}) = BFloat16
end

# Wide multiplication
Base.widemul(x::BFloat16, y::BFloat16) = Float32(x) * Float32(y)

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

# Random
import Random: rand, randn, randexp, AbstractRNG, Sampler
rand(rng::AbstractRNG, ::Sampler{BFloat16}) = convert(BFloat16, rand(rng))
randn(rng::AbstractRNG, ::Type{BFloat16}) = convert(BFloat16, randn(rng))
randexp(rng::AbstractRNG, ::Type{BFloat16}) = convert(BFloat16, randexp(rng))

# Bitstring
bitstring(x::BFloat16) = bitstring(reinterpret(UInt16, x))

# next/prevfloat
function Base.nextfloat(x::BFloat16)
    if isfinite(x)
        ui = reinterpret(UInt16,x)
        if ui < 0x8000  # positive numbers
            return reinterpret(BFloat16,ui+0x0001)
        elseif ui == 0x8000     # =-zero(T)
            return reinterpret(BFloat16,0x0001)
        else                # negative numbers
            return reinterpret(BFloat16,ui-0x0001)
        end
    else    # NaN / Inf case
        return x
    end
end

function Base.prevfloat(x::BFloat16)
    if isfinite(x)
        ui = reinterpret(UInt16,x)
        if ui == 0x0000     # =zero(T)
            return reinterpret(BFloat16,0x8001)
        elseif ui < 0x8000  # positive numbers
            return reinterpret(BFloat16,ui-0x0001)
        else                # negative numbers
            return reinterpret(BFloat16,ui+0x0001)
        end
    else    # NaN / Inf case
        return x
    end
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

