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
        throw(ArgumentError("Unknown uplo $uplo"))
    end
end

function Base.convert(::Type{onemklSide}, side::Char)
    if side == 'L'
        return ONEMKL_SIDE_LEFT
    elseif side == 'R'
        return ONEMKL_SIDE_RIGHT
    else
        throw(ArgumentError("Unknown transpose $side"))
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
