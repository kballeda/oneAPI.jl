using CEnum

@cenum onemklTranspose::UInt32 begin
    ONEMKL_TRANSPOSE_NONTRANS = 0
    ONEMKL_TRANSPOSE_TRANS = 1
    ONEMLK_TRANSPOSE_CONJTRANS = 2
end

@cenum onemklUplo::UInt32 begin
    ONEMKL_UPLO_UPPER = 0
    ONEMKL_UPLO_LOWER = 1
end

@cenum onemklSide::UInt32 begin
    ONEMKL_SIDE_LEFT = 0
    ONEMKL_SIDE_RIGHT = 1
end

function onemklSgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklSgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::Cfloat, A::ZePtr{Cfloat}, lda::Int64,
                                    B::ZePtr{Cfloat}, ldb::Int64, beta::Cfloat,
                                    C::ZePtr{Cfloat}, ldc::Int64)::Cint
end

function onemklDgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklDgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::Cdouble, A::ZePtr{Cdouble}, lda::Int64,
                                    B::ZePtr{Cdouble}, ldb::Int64, beta::Cdouble,
                                    C::ZePtr{Cdouble}, ldc::Int64)::Cint
end

function onemklCgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklCgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::ComplexF32, A::ZePtr{ComplexF32}, lda::Int64,
                                    B::ZePtr{ComplexF32}, ldb::Int64, beta::ComplexF32,
                                    C::ZePtr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklZgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::ComplexF64, A::ZePtr{ComplexF64}, lda::Int64,
                                    B::ZePtr{ComplexF64}, ldb::Int64, beta::ComplexF64,
                                    C::ZePtr{ComplexF64}, ldc::Int64)::Cint
end

function onemklSsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta,
                     c, ldc) 
    @ccall liboneapi_support.onemklSsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, m::Int64, n::Int64, alpha::Cfloat,
                                         a::ZePtr{Cfloat}, lda::Int64, b::ZePtr{Cfloat}, ldb::Int64,
                                         beta::Cfloat, c::ZePtr{Cfloat}, ldc::Int64)::Cvoid
end

function onemklDsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta,
                    c, ldc) 
    @ccall liboneapi_support.onemklSsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                        upper_lower::onemklUplo, m::Int64, n::Int64, alpha::Cdouble,
                                        a::ZePtr{Cdouble}, lda::Int64, b::ZePtr{Cdouble}, ldb::Int64,
                                        beta::Cdouble, c::ZePtr{Cdouble}, ldc::Int64)::Cvoid
end

function onemklCsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta,
                     c, ldc) 
    @ccall liboneapi_support.onemklCsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                        upper_lower::onemklUplo, m::Int64, n::Int64, alpha::ComplexF32,
                                        a::ZePtr{ComplexF32}, lda::Int64, b::ZePtr{ComplexF32},
                                        ldb::Int64, beta::ComplexF32, c::ZePtr{ComplexF32},
                                        ldc::Int64)::Cvoid
end

function onemklZsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta,
                     c, ldc) 
    @ccall liboneapi_support.onemklZsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                        upper_lower::onemklUplo, m::Int64, n::Int64, alpha::ComplexF64,
                                        a::ZePtr{ComplexF64}, lda::Int64, b::ZePtr{ComplexF64},
                                        ldb::Int64, beta::ComplexF64, c::ZePtr{ComplexF64},
                                        ldc::Int64)::Cvoid
end

function onemklSsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklSsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64, alpha::Cfloat,
                                         a::ZePtr{Cfloat}, lda::Int64, beta::Cfloat, c::ZePtr{Cfloat},
                                         ldc::Int64)::Cvoid
end

function onemklDsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklSsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64, alpha::Cdouble,
                                         a::ZePtr{Cdouble}, lda::Int64, beta::Cdouble, c::ZePtr{Cfloat},
                                         ldc::Int64)::Cvoid
end

function onemklCsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklSsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64, alpha::ComplexF32,
                                         a::ZePtr{ComplexF32}, lda::Int64, beta::ComplexF32, c::ZePtr{ComplexF32},
                                         ldc::Int64)::Cvoid
end

function onemklZsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklSsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64, alpha::ComplexF64,
                                         a::ZePtr{ComplexF64}, lda::Int64, beta::ComplexF64, c::ZePtr{ComplexF64},
                                         ldc::Int64)::Cvoid
end

function onemklSsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklSsyr2k(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                          trans::onemklTranspose, n::Int64, k::Int64, alpha::Cfloat,
                                          a::ZePtr{Cfloat}, lda::Int64, b::ZePtr{Cfloat}, ldb::Int64,
                                          beta::Cfloat, c::ZePtr{Cfloat}, ldc::Int64)::Cvoid
end

function onemklDsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklDsyr2k(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                          trans::onemklTranspose, n::Int64, k::Int64, alpha::Cdouble,
                                          a::ZePtr{Cdouble}, lda::Int64, b::ZePtr{Cdouble}, ldb::Int64,
                                          beta::Cdouble, c::ZePtr{Cdouble}, ldc::Int64)::Cvoid
end

function onemklCsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklCsyr2k(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                          trans::onemklTranspose, n::Int64, k::Int64, alpha::ComplexF32,
                                          a::ZePtr{ComplexF32}, lda::Int64, b::ZePtr{ComplexF32}, ldb::Int64,
                                          beta::ComplexF32, c::ZePtr{ComplexF32}, ldc::Int64)::Cvoid
end

function onemklZsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklZsyr2k(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                          trans::onemklTranspose, n::Int64, k::Int64, alpha::ComplexF64,
                                          a::ZePtr{ComplexF64}, lda::Int64, b::ZePtr{ComplexF64}, ldb::Int64,
                                          beta::ComplexF64, c::ZePtr{ComplexF64}, ldc::Int64)::Cvoid
end

function onemklDnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklDnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{Cdouble}, incx::Int64, 
                                result::RefOrZeRef{Cdouble})::Cvoid
end

function onemklSnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklSnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{Cfloat}, incx::Int64, 
                                result::RefOrZeRef{Cfloat})::Cvoid
end

function onemklCnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklCnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{ComplexF32}, incx::Int64, 
                                result::RefOrZeRef{Cfloat})::Cvoid
end

function onemklZnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklZnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{ComplexF64}, incx::Int64, 
                                result::RefOrZeRef{Cdouble})::Cvoid
end


function onemklDcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklDcopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{Cdouble}, incx::Int64,
                                y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklScopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklScopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{Cfloat}, incx::Int64,
                                y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklZcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklZcopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{ComplexF64}, incx::Int64,
                                y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
end

function onemklCcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklCcopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{ComplexF32}, incx::Int64,
                                y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklSamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklSamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cfloat}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklDamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklDamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cdouble}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklCamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklCamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF32}, incx::Int64,result::ZePtr{Int64})::Cvoid
end

function onemklZamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklZamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF64}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklSamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklSamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cfloat}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklDamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklDamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cdouble}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklCamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklCamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF32}, incx::Int64,result::ZePtr{Int64})::Cvoid
end

function onemklZamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklZamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF64}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklSswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklSswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{Cfloat}, incx::Cint,
                                    y::ZePtr{Cfloat}, incy::Cint)::Cvoid
end

function onemklDswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklDswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{Cdouble}, incx::Cint,
                                    y::ZePtr{Cdouble}, incy::Cint)::Cvoid
end

function onemklCswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklCswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{ComplexF32}, incx::Cint,
                                    y::ZePtr{ComplexF32}, incy::Cint)::Cvoid
end

function onemklZswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklZswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{ComplexF64}, incx::Cint,
                                    y::ZePtr{ComplexF64}, incy::Cint)::Cvoid
end