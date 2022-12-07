using LinearAlgebra

function gemm_dispatch!(C::oneStridedVecOrMat, A, B, alpha::Number=true, beta::Number=false)
    if ndims(A) > 2
        throw(ArgumentError("A has more than 2 dimensions"))
    elseif ndims(B) > 2
        throw(ArgumentError("B has more than 2 dimensions"))
    end
    mA, nA = size(A,1), size(A,2)
    mB, nB = size(B,1), size(B,2)

    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    if mA == 0 || nA == 0 || nB == 0
        if size(C) != (mA, nB)
            throw(DimensionMismatch("C has dimensions $(size(C)), should have ($mA,$nB)"))
        end
        return LinearAlgebra.rmul!(C, 0)
    end

    tA, dA = if A isa Transpose
        'T', parent(A)
    elseif A isa Adjoint
        'C', parent(A)
    else
        'N', A
    end

    tB, dB = if B isa Transpose
        'T', parent(B)
    elseif B isa Adjoint
        'C', parent(B)
    else
        'N', B
    end

    T = eltype(C)
    if T <: onemklFloat && dA isa oneStridedArray{T} && dB isa oneStridedArray{T} &&
       T != Float16 # onemklHgemm is currently not hooked-up
        gemm!(tA, tB, alpha, dA, dB, beta, C)
    else
        GPUArrays.generic_matmatmul!(C, A, B, alpha, beta)
    end
end

LinearAlgebra.BLAS.asum(x::oneStridedVecOrMat{<:onemklFloat}) = oneMKL.asum(length(x), x)

function LinearAlgebra.axpy!(alpha::Number, x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}) where T<:onemklFloat
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    oneMKL.axpy!(length(x), alpha, x, y)
end

LinearAlgebra.rmul!(x::oneStridedVecOrMat{<:onemklFloat}, k::Number) = 
	oneMKL.scal!(length(x), convert(eltype(x),k), x)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::oneStridedVecOrMat{<:onemklFloat}, k::Real) =
	invoke(rmul!, Tuple{typeof(x), Number}, x, k)
LinearAlgebra.norm(x::oneStridedVecOrMat{<:onemklFloat}) = oneMKL.nrm2(length(x), x)

function LinearAlgebra.dot(x::oneStridedArray{T}, y::oneStridedArray{T}) where T<:Union{Float32, Float64}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    oneMKL.dot(n, x, y)
end

function LinearAlgebra.dot(x::oneStridedArray{T}, y::oneStridedArray{T}) where T<:Union{ComplexF32, ComplexF64}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    oneMKL.dotc(n, x, y)
end

# level 3

@inline function LinearAlgebra.mul!(C::oneStridedVecOrMat{T}, A::Hermitian{T,<:oneStridedVecOrMat}, B::oneStridedVecOrMat{T},
                            α::Number, β::Number) where {T<:Union{Float32,Float64}}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return oneMKL.symm!('L', A.uplo, alpha, A.data, B, beta, C)
    else
        error("only supports BLAS type, got $T")
    end
end

@inline function LinearAlgebra.mul!(C::oneStridedVecOrMat{T}, A::oneStridedVecOrMat{T}, B::Hermitian{T,<:oneStridedVecOrMat},
             α::Number, β::Number) where {T<:Union{Float32,Float64}}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return oneMKL.symm!('R', B.uplo, alpha, B.data, A, beta, C)
    else
        error("only supports BLAS type, got $T")
    end
end

# triangular

## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:oneStridedVecOrMat},
                            B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B, B)
        LinearAlgebra.rmul!(A::oneStridedVecOrMat{T},
                            B::$t{T,<:oneStridedVecOrMat}) where {T<:onemklFloat} =
            oneMKL.trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A, A)

        # optimization: Base.mul! uses lmul!/rmul! with a copy (because of BLAS)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::$t{T,<:oneStridedVecOrMat},
                           B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B, X)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::oneStridedVecOrMat{T},
                           B::$t{T,<:oneStridedVecOrMat}) where {T<:onemklFloat} =
            oneMKL.trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A, X)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:oneStridedVecOrMat},
                            B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trsm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B)

        # Right division
        LinearAlgebra.rdiv!(A::oneStridedVecOrMat{T},
                            B::$t{T,<:oneStridedVecOrMat}) where {T<:onemklFloat} =
            oneMKL.trsm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A)

        # Matrix inverse
        function LinearAlgebra.inv(x::$t{T, <:oneStridedVecOrMat{T}}) where T<:onemklFloat
            out = oneStridedArray{T}(I(size(x,1)))
            $t(LinearAlgebra.ldiv!(x, out))
        end
    end
end


## adjoint/transpose multiplication ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{<:Any,<:Transpose{T,<:oneStridedVecOrMat}},
                            B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, B)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}},
                            B::oneStridedVecOrMat{T}) where {T<:Union{ComplexF32,ComplexF64}} =
            oneMKL.trmm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B, B)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}},
                            B::oneStridedVecOrMat{T}) where {T<:Union{Float32,Float64}} =
            oneMKL.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, B)

        LinearAlgebra.rmul!(A::oneStridedVecOrMat{T},
                            B::$t{<:Any,<:Transpose{T,<:oneStridedVecOrMat}}) where {T<:onemklFloat} =
            oneMKL.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, A)
        LinearAlgebra.rmul!(A::oneStridedVecOrMat{T},
                            B::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}}) where {T<:Union{ComplexF32,ComplexF64}} =
            oneMKL.trmm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A, A)
        LinearAlgebra.rmul!(A::oneStridedVecOrMat{T},
                            B::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}}) where {T<:Union{Float32,Float64}} =
            oneMKL.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, A)

        # optimization: Base.mul! uses lmul!/rmul! with a copy (because of BLAS)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::$t{<:Any,<:Transpose{T,<:oneStridedVecOrMat}},
                           B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}},
                           B::oneStridedVecOrMat{T}) where {T<:onemklComplex} =
            oneMKL.trmm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}},
                           B::oneStridedVecOrMat{T}) where {T<:onemklReal} =
            oneMKL.trmm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B, X)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::oneStridedVecOrMat{T},
                           B::$t{<:Any,<:Transpose{T,<:oneStridedVecOrMat}}) where {T<:onemklFloat} =
            oneMKL.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, X)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::oneStridedVecOrMat{T},
                           B::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}}) where {T<:Union{ComplexF32,ComplexF64}} =
            oneMKL.trmm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A, X)
        LinearAlgebra.mul!(X::oneStridedVecOrMat{T}, A::oneStridedVecOrMat{T},
                           B::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}}) where {T<:Union{Float32,Float64}} =
            oneMKL.trmm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A, X)

        # Left division
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:oneStridedVecOrMat}},
                            B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trsm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}},
                            B::oneStridedVecOrMat{T}) where {T<:Union{Float32,Float64}} =
            oneMKL.trsm!('L', $uploc, 'T', $isunitc, one(T), parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}},
                            B::oneStridedVecOrMat{T}) where {T<:Union{ComplexF32,ComplexF64}} =
            oneMKL.trsm!('L', $uploc, 'C', $isunitc, one(T), parent(parent(A)), B)

        # Right division
        LinearAlgebra.rdiv!(A::oneStridedVecOrMat{T},
                            B::$t{<:Any,<:Transpose{T,<:oneStridedVecOrMat}}) where {T<:onemklFloat} =
            oneMKL.trsm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A)
        LinearAlgebra.rdiv!(A::oneStridedVecOrMat{T},
                            B::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}}) where {T<:Union{Float32,Float64}} =
            oneMKL.trsm!('R', $uploc, 'T', $isunitc, one(T), parent(parent(B)), A)
        LinearAlgebra.rdiv!(A::oneStridedVecOrMat{T},
                            B::$t{<:Any,<:Adjoint{T,<:oneStridedVecOrMat}}) where {T<:Union{ComplexF32,ComplexF64}} =
            oneMKL.trsm!('R', $uploc, 'C', $isunitc, one(T), parent(parent(B)), A)
    end
end


function LinearAlgebra.mul!(X::oneStridedVecOrMat{T},
                            A::LowerTriangular{T,<:oneStridedVecOrMat},
                            B::UpperTriangular{T,<:oneStridedVecOrMat},
                            ) where {T<:onemklFloat}
    triu!(parent(B))
    trmm!('L', 'L', 'N', 'N', one(T), parent(A), parent(B), parent(X))
    X
end

function LinearAlgebra.mul!(X::oneStridedVecOrMat{T},
                            A::UpperTriangular{T,<:oneStridedVecOrMat},
                            B::LowerTriangular{T,<:oneStridedVecOrMat},
                            ) where {T<:onemklFloat}
    tril!(parent(B))
    trmm!('L', 'U', 'N', 'N', one(T), parent(A), parent(B), parent(X))
    X
end

for (trtype, valtype) in ((:Transpose, :onemklFloat),
                          (:Adjoint,   :onemklReal),
                          (:Adjoint,   :onemklComplex))
    @eval begin
        function LinearAlgebra.mul!(X::oneStridedVecOrMat{T},
                                    A::UpperTriangular{T,<:oneStridedVecOrMat},
                                    B::LowerTriangular{<:Any,<:$trtype{T,<:oneStridedVecOrMat}},
                                    ) where {T<:$valtype}
            # operation is reversed to avoid executing the tranpose
            triu!(parent(A))
            oneMKL.trmm!('R', 'U', 'T', 'N', one(T), parent(parent(B)), parent(A), parent(X))
            X
        end

        function LinearAlgebra.mul!(X::oneStridedVecOrMat{T},
                                    A::UpperTriangular{<:Any,<:$trtype{T,<:oneStridedVecOrMat}},
                                    B::LowerTriangular{T,<:oneStridedVecOrMat},
                                    ) where {T<:$valtype}
            tril!(parent(B))
            oneMKL.trmm!('L', 'L', 'T', 'N', one(T), parent(parent(A)), parent(B), parent(X))
            X
        end

        function LinearAlgebra.mul!(X::oneStridedVecOrMat{T},
                                    A::LowerTriangular{<:Any,<:$trtype{T,<:oneStridedVecOrMat}},
                                    B::UpperTriangular{T,<:oneStridedVecOrMat},
                                    ) where {T<:$valtype}
            triu!(parent(B))
            oneMKL.trmm!('L', 'U', 'T', 'N', one(T), parent(parent(A)), parent(B), parent(X))
            X
        end

        function LinearAlgebra.mul!(X::oneStridedVecOrMat{T},
                                    A::LowerTriangular{T,<:oneStridedVecOrMat},
                                    B::UpperTriangular{<:Any,<:$trtype{T,<:oneStridedVecOrMat}},
                                    ) where {T<:$valtype}
            # operation is reversed to avoid executing the tranpose
            tril!(parent(A))
            oneMKL.trmm!('R', 'L', 'T', 'N', one(T), parent(parent(B)), parent(A), parent(X))
            X
        end
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin
        LinearAlgebra.mul!(C::oneStridedMatrix, A::oneStridedVecOrMat, B::oneStridedVecOrMat, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneStridedMatrix, A::Transpose{<:Any, <:oneStridedVecOrMat}, B::oneStridedMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::oneStridedMatrix, B::Transpose{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::Transpose{<:Any, <:oneStridedVecOrMat}, B::Transpose{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneStridedMatrix, A::Adjoint{<:Any, <:oneStridedVecOrMat}, B::oneStridedMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::oneStridedMatrix, B::Adjoint{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::Adjoint{<:Any, <:oneStridedVecOrMat}, B::Adjoint{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneStridedMatrix, A::Transpose{<:Any, <:oneStridedVecOrMat}, B::Adjoint{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::Adjoint{<:Any, <:oneStridedVecOrMat}, B::Transpose{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
    end
end
