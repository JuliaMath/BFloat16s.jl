import Base: sign_mask, exponent_mask, significand_mask,
    exponent_one, exponent_half,
    precision, eps,
    zero, one,
    floatmax, floatmin, typemax, typemin,
    iszero, isfinite, isinf, isnan,
    (==), (!=), (<), (<=), (>=), (>),
    (+), (-), (*), (/), (\),
    signbit, sign, abs, copysign, flipsign,
    trunc, floor, ceil,
    round,
    div, fld, cld, rem, mod, divrem, fldmod,
    inv, sqrt, cbrt, hypot,
    exp, expm1, log, log1p, log10, log2,
    sin, cos, tan, csc, sec,  cot,
    asin, acos, atan, acsc, asec,  acot,
    sinh, cosh, tanh, csch, sech,  coth,
    asinh, acosh, atanh, acsch, asech,  acoth,
    pow, (^)
    # frexp, ldexp, exponent, significand


export square, cube, rsquare, rcube, rsqrt, rcbrt

using Base: IEEEFloat

primitive type BFloat16 <: AbstractFloat 16 end

BFloat16(x::UInt16) = reinterpret(BFloat16, x)
Base.UInt16(x::BFloat16) = reinterpret(UInt16, x)

BFloat16(x::UInt32) = reinterpret(BFloat16, x%UInt16)
Base.UInt32(x::BFloat16) = zero(UInt32) | reinterpret(UInt16,x)


## floating point traits ##
"""
    InfB16
Positive infinity of type [`BFloat16`](@ref).
"""
const InfB16 = BFloat16(0x7f80)

"""
    NaNB16
A not-a-number value of type [`BFloat16`](@ref).
"""
const NaNB16 = BFloat16(0x7fc0)

"""
    floatmax
Largest finite floating point value of type [`BFloat16`](@ref).
BFloat16(3.3895314e38)
"""
Base.floatmax(::Type{BFloat16}) = BFloat16(0x7f7f)

"""
    floatmin
Smallest normal floating point value of type [`BFloat16`](@ref).
BFloat16(2.3509887e-38)
"""
Base.floatmin(::Type{BFloat16}) = BFloat16(0x0100)

"""
    typemax

The highest value representable with the given type [`BFloat16`](@ref)
"""
Base.typemax(::Type{BFloat16}) = InfB16

"""
    typemin

The lowest value representable with the given type [`BFloat16`](@ref)
"""
Base.typemin(::Type{BFloat16}) = BFloat16(0xff80) # -InfB16

"""
    eps

The unit in last place (ulp) of x.
"""
Base.eps(::Type{BFloat16}) = BFloat16(0x3c00)

# zero, one
Base.zero(::Type{BFloat16}) = BFloat16(0x0000)
Base.one(::Type{BFloat16}) = BFloat16(0x3f80)


# Floating point property queries
for f in (:sign_mask, :exponent_mask, :exponent_one,
            :exponent_half, :significand_mask)
    @eval $(f)(::Type{BFloat16}) = UInt16($(f)(Float32) >> 16)
end


iszero(x::BFloat16) = UInt16(x) & ~sign_mask(BFloat16) == 0x0000
isfinite(x::BFloat16) = (UInt16(x) & exponent_mask(BFloat16)) != exponent_mask(BFloat16)
isnan(x::BFloat16) = (UInt16(x) & ~sign_mask(BFloat16)) > exponent_mask(BFloat16)
isinf(x::BFloat16) = xor(UInt16(x), 0x7f80) << 1 === zero(UInt16)

precision(::Type{BFloat16}) = 8


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
    return reinterpret(BFloat16, UInt16(h >> 16))
end

# Expansion to Float32
function Base.Float32(x::BFloat16)
    reinterpret(Float32, UInt32(reinterpret(UInt16, x)) << 16)
end

# Truncation to integer types
Base.unsafe_trunc(T::Type{<:Integer}, x::BFloat16) = unsafe_trunc(T, Float32(x))


# Conversion from Float64
function BFloat16(x::Float64)
    isnan(x) && return NaNB16
    if isinf(x) || abs(x) > floatmax(BFloat16)
        signbit(x) ? -InfB16 : InfB16
    else
        BFloat16(Float32(x))
    end
end

# Expansion to Float64
Base.Float64(x::BFloat16) = Float64(Float32(x))

BFloat16(x::Integer) = convert(BFloat16, convert(Float32, x))


# Number sign
signbit(x::BFloat16) = reinterpret(UInt16, x) & sign_mask(BFloat16) !== zero(UInt16)
sign(x::BFloat16) = signbit(x) ? reinterpret(BFloat16, 0xbf80) : one(BFloat16)

# Basic arithmetic
for f in (:+, :-, :*, :/)
    @eval ($f)(a::BFloat16, b::BFloat16) = BFloat16($(f)(Float32(a), Float32(b)))
end
-(x::BFloat16) = reinterpret(BFloat16, reinterpret(UInt16, x) ⊻ sign_mask(BFloat16))


# Floating point comparison

# isnan(x) || isnan(y)
somenans(ix::UInt16, iy::UInt16) = (ix|iy) & ~sign_mask(BFloat16) > exponent_mask(BFloat16)

# Signed zeros
twozeros(ix::UInt16, iy::UInt16) = (ix|iy)&~sign_mask(BFloat16) === zero(UInt16)


function (==)(x::BFloat16, y::BFloat16)
    ix, iy = UInt16(x), UInt16(y)
    # NaNs (isnan(x) || isnan(y))
    somenans(ix, iy) && return false
    # Signed zeros
    twozeros(ix, iy) && return true
    return ix === iy
end

function (!=)(x::BFloat16, y::BFloat16)
    ix, iy = UInt16(x), UInt16(y)
    # NaNs (isnan(x) || isnan(y))
    somenans(ix, iy) && return true
    # Signed zeros
    twozeros(ix, iy) && return false
    return ix !== iy
end

function (<)(x::BFloat16, y::BFloat16)
    ix, iy = UInt16(x), UInt16(y)
    # NaNs (isnan(x) || isnan(y))
    somenans(ix, iy) && return false
    # Signed zeros
    twozeros(ix, iy) && return false
    return ix < iy
end


function (<=)(x::BFloat16, y::BFloat16)
    ix, iy = UInt16(x), UInt16(y)
    # NaNs (isnan(x) || isnan(y))
    somenans(ix, iy) && return false
    # Signed zeros
    twozeros(ix, iy) && return true
    return ix <= iy
end

function (>)(x::BFloat16, y::BFloat16)
    ix, iy = UInt16(x), UInt16(y)
    # NaNs (isnan(x) || isnan(y))
    somenans(ix, iy) && return false
    # Signed zeros
    twozeros(ix, iy) && return false
    return ix > iy
end

function (>=)(x::BFloat16, y::BFloat16)
    ix, iy = UInt16(x), UInt16(y)
    # NaNs (isnan(x) || isnan(y))
    somenans(ix, iy) && return false
    # Signed zeros
    twozeros(ix, iy) && return true
    return ix >= iy
end


Base.widen(::Type{BFloat16}) = Float32
Base.promote_rule(::Type{Float32}, ::Type{BFloat16}) = Float32
Base.promote_rule(::Type{Float64}, ::Type{BFloat16}) = Float64
# needed to show e.g. InfB16
Base.promote_rule(::Type{BFloat16}, ::Type{Int64}) = BFloat16

# Wide multiplication
Base.widemul(a::BFloat16, b::BFloat16) = Float32(a) * Float32(b)

# string
function Base.string(x::BFloat16)
    if isinf(x)
        signbit(x) ? "-InfB16" : "InfB16"
    elseif isnan(x)
        "NaNB16"
    else
        string("BFloat16(", Float32(x), ")")
    end
end

# Showing
function Base.show(io::IO, x::BFloat16)
    hastypeinfo = BFloat16 === get(io, :typeinfo, Any)
    if isinf(x)
        print(io, signbit(x) ? "-InfB16" : "InfB16")
    elseif isnan(x)
        print(io, "NaNB16")
    else
        hastypeinfo || print(io, "BFloat16(")
        show(IOContext(io, :typeinfo=>Float32), Float32(x))
        hastypeinfo || print(io, ")")
    end
end

abs(x::BFloat16) = BFloat16(abs(Float32(x)))

function square(x::BFloat16)
    y = Float32(x)
    return BFloat16(y * y)
end

function cube(x::BFloat16)
    y = Float32(x)
    return BFloat16(y * y * y)
end

function rsquare(x::BFloat16)
    y = Float32(x)
    return BFloat16(inv(y * y))
end

function rcube(x::BFloat16)
    y = Float32(x)
    return BFloat16(inv(y * y * y))
end

rsqrt(x::BFloat16) = BFloat16(inv(sqrt(Float32(x))))
rcbrt(x::BFloat16) = BFloat16(inv(cbrt(Float32(x))))

sqrt(x::BFloat16) = BFloat16(sqrt(Float32(x)))
cbrt(x::BFloat16) = BFloat16(cbrt(Float32(x)))

floor(x::BFloat16) = BFloat16(floor(Float32(x)))
ceil(x::BFloat16) = BFloat16(ceil(Float32(x)))

exp(x::BFloat16) = BFloat16(exp(Float32(x)))
expm1(x::BFloat16) = BFloat16(expm1(Float32(x)))
log(x::BFloat16) = BFloat16(log(Float32(x)))
log1p(x::BFloat16) = BFloat16(log1p(Float32(x)))
log10(x::BFloat16) = BFloat16(log10(Float32(x)))
log2(x::BFloat16) = BFloat16(log2(Float32(x)))

sin(x::BFloat16) = BFloat16(sin(Float32(x)))
cos(x::BFloat16) = BFloat16(cos(Float32(x)))
tan(x::BFloat16) = BFloat16(tan(Float32(x)))
csc(x::BFloat16) = BFloat16(csc(Float32(x)))
sec(x::BFloat16) = BFloat16(sec(Float32(x)))
cot(x::BFloat16) = BFloat16(cot(Float32(x)))

asin(x::BFloat16) = BFloat16(asin(Float32(x)))
acos(x::BFloat16) = BFloat16(acos(Float32(x)))
atan(x::BFloat16) = BFloat16(atan(Float32(x)))
acsc(x::BFloat16) = BFloat16(acsc(Float32(x)))
asec(x::BFloat16) = BFloat16(asec(Float32(x)))
acot(x::BFloat16) = BFloat16(acot(Float32(x)))

sinh(x::BFloat16) = BFloat16(sinh(Float32(x)))
cosh(x::BFloat16) = BFloat16(cosh(Float32(x)))
tanh(x::BFloat16) = BFloat16(tanh(Float32(x)))
csch(x::BFloat16) = BFloat16(csch(Float32(x)))
sech(x::BFloat16) = BFloat16(sech(Float32(x)))
coth(x::BFloat16) = BFloat16(coth(Float32(x)))

asinh(x::BFloat16) = BFloat16(asinh(Float32(x)))
acosh(x::BFloat16) = BFloat16(acosh(Float32(x)))
atanh(x::BFloat16) = BFloat16(atanh(Float32(x)))
acsch(x::BFloat16) = BFloat16(acsch(Float32(x)))
asech(x::BFloat16) = BFloat16(asech(Float32(x)))
acoth(x::BFloat16) = BFloat16(acoth(Float32(x)))

for (A,B) in ((:BFloat16, :BFloat16),
              (:BFloat16, :Integer), (:Integer, :BFloat16),
              (:BFloat16, :IEEEFloat), (:IEEEFloat, :BFloat16))
  @eval begin
    pow(x::$A, y::$B) = BFloat16(pow(Float32(x), Float32(y)))
    (^)(x::$A, y::$B) = BFloat16(Float32(x)^Float32(y))

    rem(x::$A, y::$B) = BFloat16(rem(Float32(x), Float32(y)))
    mod(x::$A, y::$B) = BFloat16(mod(Float32(x), Float32(y)))
    div(x::$A, y::$B) = BFloat16(div(Float32(x), Float32(y)))
    cld(x::$A, y::$B) = BFloat16(cld(Float32(x), Float32(y)))
    fld(x::$A, y::$B) = BFloat16(fld(Float32(x), Float32(y)))

    divrem(x::$A, y::$B) = div(x,y), rem(x,y)
    fldmod(x::$A, y::$B) = fld(x,y), mod(x,y)
  end
end
