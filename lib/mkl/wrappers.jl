#
# Auxiliary
#

function Base.convert(::Type{onemklSide}, side::Char)
    if side == 'L'
        return ONEMKL_SIDE_LEFT
    elseif side == 'R'
        return ONEMKL_SIDE_RIGHT
    else
        throw(ArgumentError("Unknown transpose $side"))
    end
end

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

function Base.convert(::Type{onemklDiag}, diag::Char)
    if diag == 'N'
        return ONEMKL_DIAG_NONUNIT
    elseif diag == 'U'
        return ONEMKL_DIAG_UNIT
    else
        throw(ArgumentError("Unknown transpose $diag"))
    end
end

## (L3: symm) symmetric matrix-matrix and matrix-vector multiplication
for (fname, elty) in ((:onemklSsymm, :Float32),
                      (:onemklDsymm, :Float64),
                      (:onemklCsymm, :ComplexF32),
                      (:onemklZsymm, :ComplexF64))
    @eval begin
        function symm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       B::oneStridedVecOrMat{$elty},
                       beta::Number,
                       C::oneStridedVecOrMat{$elty})
            k, nA = size(A)
            if k != nA throw(DimensionMismatch("Matrix A must be square")) end
            m = side == 'L' ? k : size(B,1)
            n = side == 'L' ? size(B,2) : k
            if m != size(C,1) || n != size(C,2) || k != size(B, side == 'L' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device(A))
            $fname(sycl_queue(queue), side, uplo, m, n, alpha, A, lda, B, ldb,
                   beta, C, ldc)
            C
        end

        function symm(side::Char,
                      uplo::Char,
                      alpha::Number,
                      A::oneStridedVecOrMat{$elty},
                      B::oneStridedVecOrMat{$elty})
            symm!(side, uplo, alpha, A, B, zero($elty), similar(B))
        end

        function symm(side::Char,
                      uplo::Char,
                      A::oneStridedVecOrMat{$elty},
                      B::oneStridedVecOrMat{$elty})
            symm(side, uplo, one($elty), A, B)
        end
    end
end

## syrk
for (fname, elty) in ((:onemklDsyrk,:Float64),
                      (:onemklSsyrk,:Float32),
                      (:onemklCsyrk,:ComplexF32),
                      (:onemklZsyrk,:ComplexF64))
    @eval begin
        function syrk!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       beta::Number,
                       C::oneStridedMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("syrk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device(A))
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
        function syrk(uplo::Char,
                      trans::Char,
                      alpha::Number,
                      A::oneStridedVecOrMat)
                T = eltype(A)
                n = size(A, trans == 'N' ? 1 : 2)
                syrk!(uplo, trans, alpha, A, zero(T), similar(A, T, (n, n)))
        end
        syrk(uplo::Char, trans::Char, A::oneStridedVecOrMat) =
            syrk(uplo, trans, one(eltype(A)), A)

    end
end

## syr2k
for (fname, elty) in ((:onemklDsyr2k,:Float64),
                      (:onemklSsyr2k,:Float32),
                      (:onemklZsyr2k,:ComplexF64),
                      (:onemklCsyr2k,:ComplexF32))
    @eval begin
        function syr2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::oneStridedVecOrMat{$elty},
                        B::oneStridedVecOrMat{$elty},
                        beta::Number,
                        C::oneStridedVecOrMat{$elty})
            m, n = size(C)
            if m != n throw(DimensionMismatch("C must be square")) end
            nA = size(A, trans == 'N' ? 1 : 2)
            nB = size(B, trans == 'N' ? 1 : 2)
            if nA != n throw(DimensionMismatch("First dimension of op(A) must match C")) end
            if nB != n throw(DimensionMismatch("First dimension of op(B.') must match C")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            if k != size(B, trans == 'N' ? 2 : 1) throw(DimensionMismatch(
                "Inner dimensions of op(A) and op(B.') must match")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device(A))
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end

        function syr2k(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat,
                       B::oneStridedVecOrMat)
                T = eltype(A)
                n = size(A, trans == 'N' ? 1 : 2)
                syr2k!(uplo, trans, convert(T,alpha), A, B, zero(T), similar(A, T, (n, n)))
        end

        syr2k(uplo::Char, trans::Char, A::oneStridedVecOrMat, B::oneStridedVecOrMat) =
                syr2k(uplo, trans, one(eltype(A)), A, B)
    end
end

## (TR) Triangular matrix and vector multiplication and solution
for (mmname, smname, elty) in
        ((:onemklDtrmm, :onemklDtrsm, :Float64),
         (:onemklStrmm, :onemklStrsm, :Float32),
         (:onemklZtrmm, :onemklZtrsm, :ComplexF64),
         (:onemklCtrmm, :onemklCtrsm, :ComplexF32))
    @eval begin
        function trmm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trmm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            queue = global_queue(context(A), device(A))
            $mmname(sycl_queue(queue), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
            B
        end

        function trmm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::Number,
                      A::oneStridedMatrix{$elty},
                      B::oneStridedMatrix{$elty})
            trmm!(side, uplo, transa, diag, alpha, A, B)
        end
        function trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            queue = global_queue(context(A), device(A))
            $smname(sycl_queue(queue), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
            B
        end

        function trsm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::Number,
                      A::oneStridedMatrix{$elty},
                      B::oneStridedMatrix{$elty})
            trsm!(side, uplo, transa, diag, alpha, A, copy(B))
        end
    end
end

## hemm
for (fname, elty) in ((:onemklZhemm,:ComplexF64),
                      (:onemklChemm,:ComplexF32))
    @eval begin
        function hemm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty},
                       beta::Number,
                       C::oneStridedMatrix{$elty})
            mA, nA = size(A)
            m, n = size(B)
            mC, nC = size(C)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if ((m != mC) || (n != nC)) throw(DimensionMismatch("B and C must have same dimensions")) end
            if ((side == 'L') && (mA != m)) throw(DimensionMismatch("")) end
            if ((side == 'R') && (mA != n)) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device(A))
            $fname(sycl_queue(queue), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function hemm(uplo::Char,
                      trans::Char,
                      alpha::Number,
                      A::oneStridedMatrix{$elty},
                      B::oneStridedMatrix{$elty})
            m,n = size(B)
            hemm!( uplo, trans, alpha, A, B, zero($elty), similar(B, $elty, (m,n) ) )
        end
        hemm( uplo::Char, trans::Char, A::oneStridedMatrix{$elty}, B::oneStridedMatrix{$elty}) =
            hemm( uplo, trans, one($elty), A, B)
    end
end

## herk
for (fname, elty) in ((:onemklZherk, :ComplexF64),
                      (:onemklCherk, :ComplexF32))
    @eval begin
        function herk!(uplo::Char,
                       trans::Char,
                       alpha::Real,
                       A::oneStridedVecOrMat{$elty},
                       beta::Real,
                       C::oneStridedMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("herk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device(A))
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
        function herk(uplo::Char, trans::Char, alpha::Real, A::oneStridedVecOrMat{$elty})
            n = size(A, trans == 'N' ? 1 : 2)
            herk!(uplo, trans, alpha, A, zero(real($elty)), similar(A, $elty, (n,n)))
        end
        herk(uplo::Char, trans::Char, A::oneStridedVecOrMat{$elty}) =
            herk(uplo, trans, one(real($elty)), A)
   end
end

## her2k
for (fname, elty) in ((:onemklZher2k,:ComplexF64),
                      (:onemklCher2k,:ComplexF32))
    @eval begin
        function her2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::oneStridedVecOrMat{$elty},
                        B::oneStridedVecOrMat{$elty},
                        beta::Real,
                        C::oneStridedMatrix{$elty})
            m, n = size(C)
            if m != n throw(DimensionMismatch("C must be square")) end
            nA = size(A, trans == 'N' ? 1 : 2)
            nB = size(B, trans == 'N' ? 1 : 2)
            if nA != n throw(DimensionMismatch("First dimension of op(A) must match C")) end
            if nB != n throw(DimensionMismatch("First dimension of op(B.') must match C")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            if k != size(B, trans == 'N' ? 2 : 1)
                throw(DimensionMismatch("Inner dimensions of op(A) and op(B.') must match"))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device(A))
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function her2k(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       B::oneStridedVecOrMat{$elty})
            n = size(A, trans == 'N' ? 1 : 2)
            her2k!(uplo, trans, alpha, A, B, zero(real($elty)), similar(A, $elty, (n,n)))
        end
        her2k(uplo::Char,
              trans::Char,
              A::oneStridedVecOrMat{$elty},
              B::oneStridedVecOrMat{$elty}) = her2k(uplo, trans, one($elty), A, B)
   end
end

# level 2
## gemv
for (fname, elty) in ((:onemklSgemv, :Float32),
                      (:onemklDgemv, :Float64),
                      (:onemklCgemv, :ComplexF32),
                      (:onemklZgemv, :ComplexF64))
    @eval begin
        function gemv!(trans::Char,
                       alpha::Number,
                       a::oneStridedArray{$elty},
                       x::oneStridedArray{$elty},
                       beta::Number, 
                       y::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
             # handle trans
             m,n = size(a)
             # check dimensions
             length(x) == (trans == 'N' ? n : m) && length(y) == 
                          (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
             # compute increments
             lda = max(1,stride(a,2))
             incx = stride(x,1)
             incy = stride(y,1)
             $fname(sycl_queue(queue), trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
             y
        end

        function gemv(trans::Char,
                      alpha::Number,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            gemv!(trans, alpha, a, x, zero($elty), similar(x, $elty, size(a, (trans == 'N' ? 1 : 2))))
        end

        function gemv(trans::Char,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            gemv!(trans, one($elty), a, x, zero($elty), similar(x, $elty, size(a, (trans == 'N' ? 1 : 2))))
        end
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
                      (:onemklZhbmv,:ComplexF64))
    @eval begin

        function hbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       x::oneStridedVector{$elty},
                       beta::Number,
                       y::oneStridedVector{$elty})
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
                      A::oneStridedMatrix{$elty}, x::oneStridedVector{$elty})
            n = size(A,2)
            hbmv!(uplo, k, alpha, A, x, zero($elty), similar(x, $elty, n))
        end

        function hbmv(uplo::Char, k::Integer, A::oneStridedMatrix{$elty},
                      x::oneStridedVector{$elty})
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
## axpy primitive
for (fname, elty) in 
        ((:onemklDaxpy,:Float64),
         (:onemklSaxpy,:Float32),
         (:onemklHaxpy,:Float16),
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

function axpy!(n::Integer, 
            alpha::Number,
            x::oneStridedArray{ComplexF16},
            y::oneStridedArray{ComplexF16})
    wide_x = widen.(x)
    wide_y = widen.(y)
    axpy!(n, alpha, wide_x, wide_y)
    thin_y = convert(typeof(y), wide_y)
    copyto!(y, thin_y)
    return y
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

## dot
for (jname, fname, elty) in
        ((:dot, :onemklSdot,:Float32),
         (:dot, :onemklDdot,:Float64),
         (:dotc, :onemklCdotc, :ComplexF32),
         (:dotc, :onemklZdotc, :ComplexF64),
         (:dotu, :onemklCdotu, :ComplexF32),
         (:dotu, :onemklZdotu, :ComplexF64))
    @eval begin
        function $jname(n::Integer,
                         x::oneStridedArray{$elty},
                         y::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            result = oneArray{$elty}([0]);
            $fname(sycl_queue(queue), n, x, stride(x,1), y, stride(y,1), result)
            res = Array(result)
            return res[1]
        end
    end
end

# level 2
# sbmv, symmetric banded matrix-vector multiplication
for (fname, elty) in ((:onemklSsbmv, :Float32),
                      (:onemklDsbmv, :Float64))
    @eval begin
        function sbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       a::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty},
                       beta::Number,
                       y::oneStridedVecOrMat{$elty})
            m, n = size(a)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            queue = global_queue(context(x), device(x))
            lda = max(1, stride(a,2))
            incx = stride(x,1)
            incy = stride(y,1)
            alpha = $elty(alpha)
            beta = $elty(beta)
            $fname(sycl_queue(queue), uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
            y
        end

        function sbmv(uplo::Char, k::Integer, alpha::Number,
                      a::oneStridedArray{$elty}, x::oneStridedArray{$elty})
            n = size(a,2)
            sbmv!(uplo, k, alpha, a, x, zero($elty), similar(x, $elty, n))
        end

        function sbmv(uplo::Char, k::Integer, a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            sbmv(uplo, k, one($elty), a, x)
        end
    end
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

# level 2
# ger
for (fname, elty) in ((:onemklSger, :Float32),
                      (:onemklDger, :Float64),
                      (:onemklCgerc, :ComplexF32),
                      (:onemklZgerc, :ComplexF64))
    @eval begin
        function ger!(alpha::Number,
                      x::oneStridedVecOrMat{$elty},
                      y::oneStridedVecOrMat{$elty},
                      a::oneStridedVecOrMat{$elty})
            m,n = size(a)
            m == length(x) || throw(DimensionMismatch(""))
            n == length(y) || throw(DimensionMismatch(""))
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), m, n, alpha, x, stride(x,1), y, stride(y,1), a, max(1,stride(a,2)))
            a
        end
    end
end

#symv
for (fname, elty) in ((:onemklSsymv,:Float32),
                      (:onemklDsymv,:Float64))
    # Note that the complex symv are not BLAS but auiliary functions in LAPACK
    @eval begin
        function symv!(uplo::Char,
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

        function symv(uplo::Char, alpha::Number, A::oneStridedVecOrMat{$elty}, x::oneStridedVecOrMat{$elty})
                symv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function symv(uplo::Char, A::oneStridedVecOrMat{$elty}, x::oneStridedVecOrMat{$elty})
            symv(uplo, one($elty), A, x)
        end

    end
end

# syr
for (fname, elty) in ((:onemklSsyr,:Float32),
                      (:onemklDsyr,:Float64))
    @eval begin
        function syr!(uplo::Char,
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
function copy!(n::Integer, x::oneStridedArray{T}, y::oneStridedArray{T}) where {T <: Union{Float16, ComplexF16}}
    copyto!(y,x)
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

# level 2
# gbmv
for (fname, elty) in ((:onemklSgbmv, :Float32),
                      (:onemklDgbmv, :Float64),
                      (:onemklCgbmv, :ComplexF32),
                      (:onemklZgbmv, :ComplexF64))
    @eval begin
        function gbmv!(trans::Char,
                       m::Integer,
                       kl::Integer,
                       ku::Integer,
                       alpha::Number,
                       a::oneStridedArray{$elty},
                       x::oneStridedArray{$elty},
                       beta::Number,
                       y::oneStridedArray{$elty})
            n = size(a,2)
            length(x) == (trans == 'N' ? n : m) && length(y) == 
                         (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            queue = global_queue(context(x), device(x))
            lda = max(1, stride(a,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(sycl_queue(queue), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)
            y
        end

        function gbmv(trans::Char,
                      m::Integer, 
                      kl::Integer,
                      ku::Integer,
                      alpha::Number,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            n = size(a,2)
            leny = trans == 'N' ? m : n
            queue = global_queue(context(x), device(x))
            gbmv!(trans, m, kl, ku, alpha, a, x, zero($elty), similar(x, $elty, leny))   
        end

        function gbmv(trans::Char,
                      m::Integer,
                      kl::Integer,
                      ku::Integer,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            gbmv(trans, m, kl, ku, one($elty), a, x)
        end
    end
end

# tbmv
### tbmv, (TB) triangular banded matrix-vector multiplication
for (fname, elty) in ((:onemklStbmv,:Float32),
                      (:onemklDtbmv,:Float64),
                      (:onemklCtbmv,:ComplexF32),
                      (:onemklZtbmv,:ComplexF64))
    @eval begin
        function tbmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, trans, diag, n, k, A, lda, x, incx)
            x
        end

        function tbmv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      k::Integer,
                      A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            tbmv!(uplo, trans, diag, k, A, copy(x))
        end
    end
end

### trmv, Triangular matrix-vector multiplication
for (fname, elty) in ((:onemklStrmv, :Float32),
                      (:onemklDtrmv, :Float64),
                      (:onemklCtrmv, :ComplexF32),
                      (:onemklZtrmv, :ComplexF64))
    @eval begin
        function trmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, trans, diag, n, A, lda, x, incx)
            x
        end

        function trmv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            trmv!(uplo, trans, diag, A, copy(x))
        end
    end
end

### trsv, Triangular matrix-vector solve
for (fname, elty) in ((:onemklStrsv, :Float32),
                      (:onemklDtrsv, :Float64),
                      (:onemklCtrsv, :ComplexF32),
                      (:onemklZtrsv, :ComplexF64))
    @eval begin
        function trsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, trans, diag, n, A, lda, x, incx)
            x
        end
        function trsv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            trsv!(uplo, trans, diag, A, copy(x))
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
