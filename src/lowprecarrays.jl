import LinearAlgebra

"""
LowPrecArray

An n-dimensional array that behaves essentially like a Float32 array for all
scalar operations, but matmuls are performed in BFloat16 with Float32
accumulation. This is intended to match the behavior of TPUs.
"""
struct LowPrecArray{N, A<:AbstractArray{Float32}} <: AbstractArray{Float32, N}
    storage::A
end
LowPrecArray(A::AbstractArray{Float32, N}) where {N} =
    LowPrecArray{N, typeof(A)}(A)

Base.similar(a::LowPrecArray, b::Type{Float32}, dims::Dims) =
    LowPrecArray(similar(a.storage, b, dims))
Base.size(a::LowPrecArray) = size(a.storage)
Base.getindex(A::LowPrecArray, inds...) = getindex(A.storage, inds...)
Base.setindex!(A::LowPrecArray, args...) = setindex!(A.storage, args...)

# A BFloat16 whose multiplication is a widemul
struct ExpandingBFloat16
    a::BFloat16
end
ExpandingBFloat16(a::Float32) = ExpandingBFloat16(BFloat16(a))
*(a::ExpandingBFloat16, b::ExpandingBFloat16) = widemul(a.a, b.a)

struct MatMulView{A} <: AbstractArray{ExpandingBFloat16, 2}
    a::A
end
Base.size(a::MatMulView) = size(a.a)
Base.getindex(A::MatMulView, inds...) = ExpandingBFloat16(getindex(A.a, inds...))
# setindex! deliberately not defined

import LinearAlgebra: generic_matmatmul!
function generic_matmatmul!(C::AbstractMatrix{Float32}, ta, tb,
    A::Union{LowPrecArray, AbstractMatrix{BFloat16}},
    B::Union{LowPrecArray, AbstractMatrix{BFloat16}})
    generic_matmatmul!(C, ta, tb,
        MatMulView(A), MatMulView(B))
end

# For now do gemv! in Float32 precision - we'll have to evaluate if this makes
# sense on the real hardware (which would have to do this on the vector units)
function LinearAlgebra.BLAS.gemv!(trans::AbstractChar, alpha::Float32,
                                  A::AbstractVecOrMat{Float32}, X::AbstractVector{Float32},
                                  beta::Float32, Y::LowPrecArray)
    LinearAlgebra.BLAS.gemv!(trans, alpha, A, X, beta, Y.storage)
end
