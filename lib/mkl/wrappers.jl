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

function Base.convert(::Type{onemklUplo}, uplo::Char)
    if uplo == 'U'
        return ONEMKL_UPLO_UPPER
    elseif uplo == 'L'
        return ONEMKL_UPLO_LOWER
    else
        throw(ArgumentError("Unknown transpose $uplo"))
    end
end


### hemv
for (fname, elty) in ((:onemklChemv,:ComplexF32),
                      (:onemklZhemv,:ComplexF64))
    @eval begin
        function hemv!(uplo::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty},
                       beta::Number,
                       y::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            y
        end

        function hemv(uplo::Char, alpha::Number, A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            hemv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function hemv(uplo::Char, A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            hemv(uplo, one($elty), A, x)
        end
    end
end

### hbmv, (HB) Hermitian banded matrix-vector multiplication
for (fname, elty) in ((:onemklChbmv,:ComplexF32),
                      (:cublasZhbmv,:ComplexF64))
    @eval begin

        function hbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty},
                       beta::Number,
                       y::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            y
        end

        function hbmv(uplo::Char, k::Integer, alpha::Number,
                      A::oneStridedVecOrMat{$elty}, x::oneStridedVecOrMat{$elty})
            n = size(A,2)
            hbmv!(uplo, k, alpha, A, x, zero($elty), similar(x, $elty, n))
        end

        function hbmv(uplo::Char, k::Integer, A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            hbmv(uplo, k, one($elty), A, x)
        end

    end
end

### her
for (fname, elty) in ((:onemklCher,:ComplexF32),
                      (:onemklZher,:ComplexF64))
    @eval begin
        function her!(uplo::Char,
                      alpha::Number,
                      x::oneStridedVecOrMat{$elty},
                      A::oneStridedVecOrMat{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, alpha, x, incx, A, lda)
            A
        end
    end
end

### her2
for (fname, elty) in ((:onemklCher2,:ComplexF32),
                      (:onemklZher2,:ComplexF64))
    @eval begin
        function her2!(uplo::Char,
                      alpha::Number,
                      x::oneStridedVecOrMat{$elty},
                      y::oneStridedVecOrMat{$elty},
                      A::oneStridedVecOrMat{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            length(y) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, alpha, x, incx, y, incy, A, lda)
            A
        end
    end
end

# level 1
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
                       x::StridedArray{$elty},
                       y::StridedArray{$elty})
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, x, stride(x, 1), y, stride(y, 1))
            y
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
