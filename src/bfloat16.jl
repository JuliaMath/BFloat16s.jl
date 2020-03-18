primitive type BFloat16 <: AbstractFloat 16 end

# Floating point property queries
for f in (:sign_mask, :exponent_mask, :exponent_one,
            :exponent_half, :significand_mask)
    @eval $(f)(::Type{BFloat16}) = UInt16($(f)(Float32) >> 16)
end

iszero(x::BFloat16) = reinterpret(UInt16, x) & ~sign_mask(BFloat16) == 0x0000
isfinite(x::BFloat16) = (reinterpret(UInt16,x) & exponent_mask(BFloat16)) != exponent_mask(BFloat16)
isnan(x::BFloat16) = (reinterpret(UInt16,x) & ~sign_mask(BFloat16)) > exponent_mask(BFloat16)

precision(::Type{BFloat16}) = 8
one(::Type{BFloat16}) = reinterpret(BFloat16,0x3f80)
zero(::Type{BFloat16}) = reinterpret(BFloat16,0x0000)

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
Base.trunc(::Type{BFloat16}, x::Float32) = reinterpret(BFloat16,
        (reinterpret(UInt32, x) >> 16) % UInt16
    )

# round to integer value via conversion
round(x::BFloat16, r::RoundingMode{:Up}) = BFloat16(ceil(Float32(x)))
round(x::BFloat16, r::RoundingMode{:Down}) = BFloat16(floor(Float32(x)))
round(x::BFloat16, r::RoundingMode{:Nearest}) = BFloat16(round(Float32(x)))

# conversion to integer
Int64(x::BFloat16) = Int64(Float32(x))
Int32(x::BFloat16) = Int32(Float32(x))
Int16(x::BFloat16) = Int16(Float32(x))
Int8(x::BFloat16) = Int8(Float32(x))
UInt64(x::BFloat16) = UInt64(Float32(x))
UInt32(x::BFloat16) = UInt32(Float32(x))
UInt16(x::BFloat16) = UInt16(Float32(x))
UInt8(x::BFloat16) = UInt8(Float32(x))

# Conversion to and from Float32
function BFloat16(x::Float32)
    isnan(x) && return NaNB16
    # Round to nearest even (matches TensorFlow and our convention for
    # rounding to lower precision floating point types).
    h = reinterpret(UInt32, x)
    h += 0x7fff + ((h >> 16) & 1)
    return reinterpret(BFloat16, (h >> 16) % UInt16)
end

# Expansion to Float32 - no rounding applied
Float32(x::BFloat16) = reinterpret(Float32, UInt32(reinterpret(UInt16, x)) << 16)

# Conversions
BFloat16(x::Float64) = BFloat16(Float32(x))
BFloat16(x::Float16) = BFloat16(Float32(x))
BFloat16(x::Integer) = BFloat16(Float32(x))

Float64(x::BFloat16) = Float64(Float32(x))
Float16(x::BFloat16) = Float16(Float32(x))

# Truncation to integer types
Base.unsafe_trunc(T::Type{<:Integer}, x::BFloat16) = unsafe_trunc(T, Float32(x))

# Basic arithmetic
for f in (:+, :-, :*, :/, :^)
    @eval ($f)(x::BFloat16, y::BFloat16) = BFloat16($(f)(Float32(x), Float32(y)))
end

-(x::BFloat16) = reinterpret(BFloat16, reinterpret(UInt16, x) ⊻ sign_mask(BFloat16))

# bit-wise & with ~sign_mask
abs(x::BFloat16) = reinterpret(BFloat16, reinterpret(UInt16, x) & 0x7fff)

for func in (:sin,:cos,:tan,:asin,:acos,:atan,:sinh,:cosh,:tanh,:asinh,:acosh,
             :atanh,:exp,:exp2,:exp10,:expm1,:log,:log2,:log10,:sqrt,:cbrt,:log1p)
    @eval begin
        Base.$func(a::BFloat16) = BFloat16($func(Float32(a)))
    end
end

for func in (:atan,:hypot)
    @eval begin
        $func(a::BFloat16,b::BFloat16) = BFloat16($func(Float32(a),Float32(b)))
    end
end

# Floating point comparison
function Base.:(==)(x::BFloat16, y::BFloat16)
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

function Base.:(<)(x::BFloat16, y::BFloat16)
	return Float32(x) < Float32(y)
end

function Base.:(<=)(x::BFloat16, y::BFloat16)
	return Float32(x) <= Float32(y)
end

function Base.:(>)(x::BFloat16, y::BFloat16)
	return Float32(x) > Float32(y)
end

function Base.:(>=)(x::BFloat16, y::BFloat16)
	return Float32(x) >= Float32(y)
end

widen(::Type{BFloat16}) = Float32

promote_rule(::Type{Float32}, ::Type{BFloat16}) = Float32
promote_rule(::Type{Float64}, ::Type{BFloat16}) = Float64

for t in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    @eval promote_rule(::Type{BFloat16}, ::Type{$t}) = BFloat16
end

# Wide multiplication
widemul(x::BFloat16, y::BFloat16) = Float32(x) * Float32(y)

# Showing
function show(io::IO, x::BFloat16)
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

bitstring(x::BFloat16) = bitstring(reinterpret(UInt16,x))

function bitstring(x::BFloat16,mode::Symbol)
    if mode == :split	# split into sign, exponent, signficand
        s = bitstring(x)
		return "$(s[1]) $(s[2:9]) $(s[10:end])"
    else
        return bitstring(x)
    end
end

function nextfloat(x::BFloat16)
    if isfinite(x)
		ui = reinterpret(UInt16,x)
		if ui < 0x8000	# positive numbers
			return reinterpret(BFloat16,ui+0x0001)
		elseif ui == 0x8000		# =-zero(T)
			return reinterpret(BFloat16,0x0001)
		else				# negative numbers
			return reinterpret(BFloat16,ui-0x0001)
		end
	else	# NaN / Inf case
		return x
	end
end

function prevfloat(x::BFloat16)
    if isfinite(x)
		ui = reinterpret(UInt16,x)
		if ui == 0x0000		# =zero(T)
			return reinterpret(BFloat16,0x8001)
		elseif ui < 0x8000	# positive numbers
			return reinterpret(BFloat16,ui-0x0001)
		else				# negative numbers
			return reinterpret(BFloat16,ui+0x0001)
		end
	else	# NaN / Inf case
		return x
	end
end
