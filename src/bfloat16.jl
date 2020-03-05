primitive type BFloat16 <: AbstractFloat 16 end			# deterministic rounding
primitive type BFloat16sr <: AbstractFloat 16 end		# stochastic rounding

# Floating point property queries
for f in (:sign_mask, :exponent_mask, :exponent_one,
            :exponent_half, :significand_mask)
    @eval $(f)(::Type{BFloat16}) = UInt16($(f)(Float32) >> 16)
end

iszero(x::Union{BFloat16,BFloat16sr}) = reinterpret(UInt16, x) & ~sign_mask(BFloat16) == 0x0000
isfinite(x::Union{BFloat16,BFloat16sr}) = (reinterpret(UInt16,x) & exponent_mask(BFloat16)) != exponent_mask(BFloat16)
isnan(x::Union{BFloat16,BFloat16sr}) = (reinterpret(UInt16,x) & ~sign_mask(BFloat16)) > exponent_mask(BFloat16)

precision(::Type{BFloat16}) = 8
precision(::Type{BFloat16sr}) = 8

# floating point traits #
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

# Truncation from Float32
Base.uinttype(::Type{BFloat16}) = UInt16
Base.uinttype(::Type{BFloat16sr}) = UInt16
Base.trunc(::Type{BFloat16}, x::Float32) = reinterpret(BFloat16,
        (reinterpret(UInt32, x) >> 16) % UInt16
    )
Base.trunc(::Type{BFloat16sr}, x::Float32) = reinterpret(BFloat16sr,
        (reinterpret(UInt32, x) >> 16) % UInt16
    )

# round to integer value via conversion
Base.round(x::BFloat16, r::RoundingMode{:Up}) = BFloat16(ceil(Float32(x)))
Base.round(x::BFloat16, r::RoundingMode{:Down}) = BFloat16(floor(Float32(x)))
Base.round(x::BFloat16, r::RoundingMode{:Nearest}) = BFloat16(round(Float32(x)))
Base.Int64(x::BFloat16) = Int64(Float32(x))

# same for BFloat16sr, but do not apply stochastic rounding to avoid InexactError
Base.round(x::BFloat16, r::RoundingMode{:Up}) = BFloat16(ceil(Float32(x)))
Base.round(x::BFloat16, r::RoundingMode{:Down}) = BFloat16(floor(Float32(x)))
Base.round(x::BFloat16, r::RoundingMode{:Nearest}) = BFloat16(round(Float32(x)))
Base.Int64(x::BFloat16) = Int64(Float32(x))

# Conversion from Float32
function BFloat16(x::Float32)
    isnan(x) && return NaNB16
    # Round to nearest even (matches TensorFlow and our convention for
    # rounding to lower precision floating point types).
    h = reinterpret(UInt32, x)
    h += 0x7fff + ((h >> 16) & 1)
    return reinterpret(BFloat16, (h >> 16) % UInt16)
end

"""
    epsBF16
Machine epsilon for BFloat16 as Float32.
"""
const epsBF16 = 0.0078125f0

# Conversion from Float32 with distance proportional stochastic rounding
function BFloat16sr(x::Float32)
    isnan(x) && return NaNB16

	# stochastic rounding, e is the base 2 exponent of x (sign and signficand set to zero)
	#TODO test other RNG, Xorshift64?
	#TODO there's currently a 0.001% chance to produce an overflow with bfloatmax::Float32
	e = reinterpret(Float32,reinterpret(UInt32,x) & exponent_mask(Float32))
	x += epsBF16*e*(rand(Float32)-0.5f0)

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

# Conversion from Float64
function BFloat16sr(x::Float64)
	BFloat16sr(Float32(x))
end

# Conversion from Integer
function BFloat16(x::Integer)
	convert(BFloat16, convert(Float32, x))
end

function BFloat16sr(x::Integer)
	convert(BFloat16sr, convert(Float32, x))
end

# Expansion to Float32 - no rounding applied
function Base.Float32(x::Union{BFloat16,BFloat16sr})
    reinterpret(Float32, UInt32(reinterpret(UInt16, x)) << 16)
end

# Expansion to Float64
function Base.Float64(x::Union{BFloat16,BFloat16sr})
    Float64(Float32(x))
end

# Truncation to integer types
Base.unsafe_trunc(T::Type{<:Integer}, x::Union{BFloat16,BFloat16sr}) = unsafe_trunc(T, Float32(x))

# Basic arithmetic
for f in (:+, :-, :*, :/, :^)
    @eval ($f)(x::BFloat16, y::BFloat16) = BFloat16($(f)(Float32(x), Float32(y)))
	@eval ($f)(x::BFloat16sr, y::BFloat16sr) = BFloat16sr($(f)(Float32(x), Float32(y)))
end

-(x::BFloat16) = reinterpret(BFloat16, reinterpret(UInt16, x) ⊻ sign_mask(BFloat16))
-(x::BFloat16sr) = reinterpret(BFloat16sr, reinterpret(UInt16, x) ⊻ sign_mask(BFloat16))

# bit-wise & with ~sign_mask
abs(x::BFloat16) = reinterpret(BFloat16, reinterpret(UInt16, x) & 0x7fff)
abs(x::BFloat16sr) = reinterpret(BFloat16sr, reinterpret(UInt16, x) & 0x7fff)

Base.sqrt(x::BFloat16) = BFloat16(sqrt(Float32(x)))
Base.sqrt(x::BFloat16sr) = BFloat16sr(sqrt(Float32(x)))

# Floating point comparison
function Base.:(==)(x::T, y::T) where {T<:Union{BFloat16,BFloat16sr}}
    ix = reinterpret(UInt16, x)
    iy = reinterpret(UInt16, y)
    # NaNs (isnan(x) || isnan(y))
    if (ix|iy)&~sign_mask(T) > exponent_mask(T)
        return false
    end
    # Signed zeros
    if (ix|iy)&~sign_mask(T) == 0
        return true
    end
    return ix == iy
end

function Base.:(<)(x::T, y::T) where {T<:Union{BFloat16,BFloat16sr}}
	return Float32(x) < Float32(y)
end

function Base.:(<=)(x::T, y::T) where {T<:Union{BFloat16,BFloat16sr}}
	return Float32(x) <= Float32(y)
end

function Base.:(>)(x::T, y::T) where {T<:Union{BFloat16,BFloat16sr}}
	return Float32(x) > Float32(y)
end

function Base.:(>=)(x::T, y::T) where {T<:Union{BFloat16,BFloat16sr}}
	return Float32(x) >= Float32(y)
end

Base.widen(::Type{BFloat16}) = Float32
Base.widen(::Type{BFloat16sr}) = Float32

Base.promote_rule(::Type{Float32}, ::Type{BFloat16}) = Float32
Base.promote_rule(::Type{Float64}, ::Type{BFloat16}) = Float64

Base.promote_rule(::Type{Float32}, ::Type{BFloat16sr}) = Float32
Base.promote_rule(::Type{Float64}, ::Type{BFloat16sr}) = Float64

Base.promote_rule(::Type{BFloat16}, ::Type{BFloat16sr}) = BFloat16

for t in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval Base.promote_rule(::Type{BFloat16}, ::Type{$t}) = BFloat16
	@eval Base.promote_rule(::Type{BFloat16sr}, ::Type{$t}) = BFloat16sr
end

# Wide multiplication
Base.widemul(x::BFloat16, y::BFloat16) = Float32(x) * Float32(y)
Base.widemul(x::BFloat16sr, y::BFloat16sr) = Float32(x) * Float32(y)

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

# Showing
function Base.show(io::IO, x::BFloat16sr)
    hastypeinfo = BFloat16 === get(io, :typeinfo, Any)
    if isinf(x)
        print(io, x < 0 ? "-InfB16" : "InfB16")
    elseif isnan(x)
        print(io, "NaNB16")
    else
        hastypeinfo || print(io, "BFloat16sr(")
        show(IOContext(io, :typeinfo=>Float32), Float32(x))
        hastypeinfo || print(io, ")")
    end
end

bitstring(x::Union{BFloat16,BFloat16sr}) = bistring(reinterpret(UInt16,x))
