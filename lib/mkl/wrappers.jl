#
# Auxiliary
#

function Base.convert(::Type{onemklTranspose}, trans::Char)
    if trans == 'N'
        return ONEMKL_TRANSPOSE_NONTRANS
    elseif trans == 'T'
        return ONEMKL_TRANSPOSE_TRANS
    elseif trans == 'C'
        return ONEMLK_TRANSPOSE_CONJTRANS
    else
        throw(ArgumentError("Unknown transpose $trans"))
    end
end

# level 1
## axpy primitive
for (fname, elty) in 
        ((:onemklDaxpy,:Float64),
         (:onemklSaxpy,:Float32),
         (:onemklZaxpy,:ComplexF64),
         (:onemklCaxpy,:ComplexF32))
    @eval begin
        function axpy!(n::Integer,
                       alpha::Number,
                       x::oneStridedArray{$elty},
                       y::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            alpha = $elty(alpha)
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1), y, stride(y,1))
            y
        end
    end
end

## scal
for (fname, elty) in
    ((:onemklDscal,:Float64),
     (:onemklSscal,:Float32),
     (:onemklZscal,:ComplexF64),
     (:onemklCscal,:ComplexF32))
    @eval begin
        function scal!(n::Integer,
                       alpha::$elty,
                       x::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1))
            x
        end
    end
end

## nrm2
for (fname, elty, ret_type) in
    ((:onemklDnrm2, :Float64,:Float64),
     (:onemklSnrm2, :Float32,:Float32),
     (:onemklCnrm2, :ComplexF32,:Float32),
     (:onemklZnrm2, :ComplexF64,:Float64))
    @eval begin
        function nrm2(n::Integer, x::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            result = oneArray{$ret_type}([0]);
            $fname(sycl_queue(queue), n, x, stride(x,1), result)            
            res = Array(result)
            return res[1]
        end
    end
end

# create a batch of pointers in device memory from a batch of device arrays
@inline function unsafe_batch(batch::Vector{<:oneStridedVecOrMat{T}}) where {T}
    ptrs = pointer.(batch)
    return oneStridedVecOrMat(ptrs)
end

for (fname, elty, celty) in ((:onemklCsscal, :Float32, :ComplexF32),
                             (:onemklZdscal, :Float64, :ComplexF64))
    @eval begin
        function scal!(n::Integer, 
                       alpha::$elty,
                       x::oneStridedArray{$celty})
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1))
        end
    end
end
#
# BLAS
#

# level 1
## copy
for (fname, elty) in
        ((:onemklDcopy,:Float64),
         (:onemklScopy,:Float32),
         (:onemklZcopy,:ComplexF64),
         (:onemklCcopy,:ComplexF32))
    @eval begin
        function copy!(n::Integer,
                       x::oneStridedArray{$elty},
                       y::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, x, stride(x, 1), y, stride(y, 1))
            y
        end
    end
end

## asum
for (fname, elty, ret_type) in 
    ((:onemklSasum, :Float32, :Float32),
     (:onemklDasum, :Float64, :Float64),
     (:onemklCasum, :ComplexF32, :Float32),
     (:onemklZasum, :ComplexF64, :Float64))
    @eval begin
        function asum(n::Integer,
                      x::oneStridedArray{$elty})
            result = oneArray{$ret_type}([0])
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, x, stride(x, 1), result)
            res = Array(result)
            return res[1]
        end
    end
end

## iamax
for (fname, elty) in
    ((:onemklDamax,:Float64),
     (:onemklSamax,:Float32),
     (:onemklZamax,:ComplexF64),
     (:onemklCamax,:ComplexF32))
    @eval begin
        function iamax(x::oneStridedArray{$elty})
            n = length(x)
            queue = global_queue(context(x), device(x))
            result = oneArray{Int64}([0]);
            $fname(sycl_queue(queue), n, x, stride(x, 1), result)
            return Array(result)[1]+1
        end
    end
end

## iamin
for (fname, elty) in
    ((:onemklDamin,:Float64),
     (:onemklSamin,:Float32),
     (:onemklZamin,:ComplexF64),
     (:onemklCamin,:ComplexF32))
    @eval begin
        function iamin(x::StridedArray{$elty})
            n = length(x)
            result = oneArray{Int64}([0]);
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue),n, x, stride(x, 1), result)
            return Array(result)[1]+1
        end
    end
end

## swap
for (fname, elty) in ((:onemklSswap,:Float32),
    (:onemklDswap,:Float64),
    (:onemklCswap,:ComplexF32),
    (:onemklZswap,:ComplexF64))
    @eval begin
        function swap!(n::Integer,
            x::oneStridedArray{$elty},
            y::oneStridedArray{$elty})
            # Assuming both memory allocated on same device & context
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, x, stride(x, 1), y, stride(y, 1))
            x, y
        end
    end
end

# level 3

for (fname, elty) in
        ((:onemklDgemm,:Float64),
         (:onemklSgemm,:Float32),
         (:onemklHgemm, :Float16),
         (:onemklZgemm,:ComplexF64),
         (:onemklCgemm,:ComplexF32))
    @eval begin
        function gemm!(transA::Char,
                       transB::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       B::oneStridedVecOrMat{$elty},
                       beta::Number,
                       C::oneStridedVecOrMat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end

            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))

            device(A) == device(B) == device(C) || error("Multi-device GEMM not supported")
            context(A) == context(B) == context(C) || error("Multi-context GEMM not supported")
            queue = global_queue(context(A), device(A))

            alpha = $elty(alpha)
            beta = $elty(beta)

            $fname(sycl_queue(queue), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end

        function gemm(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::oneStridedVecOrMat{$elty},
                      B::oneStridedVecOrMat{$elty})
            gemm!(transA, transB, alpha, A, B, zero($elty),
                  similar(B, $elty, (size(A, transA == 'N' ? 1 : 2),
                                     size(B, transB == 'N' ? 2 : 1))))
        end

        function gemm(transA::Char,
                      transB::Char,
                      A::oneStridedVecOrMat{$elty},
                      B::oneStridedVecOrMat{$elty})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end

## (GE) general matrix-matrix multiplication batched
for (fname, elty) in
        ((:onemklDgemm_batch,:Float64),
         (:onemklSgemm_batch,:Float32),
         (:onemklZgemm_batch,:ComplexF64),
         (:onemklCgemm_batch,:ComplexF32))
    @eval begin
        function gemm_batched!(transA::Char,
                               transB::Char,
                               alpha::Number,
                               A::Vector{<:oneStridedVecOrMat{$elty}},
                               B::Vector{<:oneStridedVecOrMat{$elty}},
                               beta::Number,
                               C::Vector{<:oneStridedVecOrMat{$elty}})
            if length(A) != length(B) || length(A) != length(C)
                throw(DimensionMismatch(""))
            end
            for (As,Bs,Cs) in zip(A,B,C)
                m = size(As, transA == 'N' ? 1 : 2)
                k = size(As, transA == 'N' ? 2 : 1)
                n = size(Bs, transB == 'N' ? 2 : 1)
                if m != size(Cs,1) || n != size(Cs,2) || k != size(Bs, transB == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end

            m = size(A[1], transA == 'N' ? 1 : 2)
            k = size(A[1], transA == 'N' ? 2 : 1)
            n = size(B[1], transB == 'N' ? 2 : 1)
            lda = max(1,stride(A[1],2))
            ldb = max(1,stride(B[1],2))
            ldc = max(1,stride(C[1],2))
            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)
            Cptrs = unsafe_batch(C)
            queue = global_queue(context(A), device(A))
            $fname(sycl_queue(queue), transA, transB, m, n, k, alpha, Aptrs, lda, 0, Bptrs,
                   ldb, 0, beta, Cptrs, ldc, 0, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)
            C
        end

        function gemm_batched(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::Vector{<:oneStridedVecOrMat{$elty}},
                      B::Vector{<:oneStridedVecOrMat{$elty}})
            C = oneStridedVecOrMat{$elty}[similar( B[1], $elty, (size(A[1], transA == 'N' ? 1 : 2),size(B[1], transB == 'N' ? 2 : 1))) for i in 1:length(A)]
            gemm_batched!(transA, transB, alpha, A, B, zero($elty), C )
        end

        function gemm_batched(transA::Char,
                            transB::Char,
                            A::Vector{<:oneStridedVecOrMat{$elty}},
                            B::Vector{<:oneStridedVecOrMat{$elty}})
            gemm_batched(transA, transB, one($elty), A, B)
        end
    end
end