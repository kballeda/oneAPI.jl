using oneAPI
using MKL
using LinearAlgebra
function main(; n = 1, T = ComplexF64)
    x = ones(T, n)
    y = ones(T, n)
    #c = rand(real(T))
    #s = T(sqrt(1 - c^2)) 
    c = 0.8187587885612081
    s = T(sqrt(1-c^2))
    @show x
    @show y

    @test c*c + s*conj(s) ≈ 1
    d_x = oneArray(x)
    d_y = oneArray(y)
    
    incx = stride(x, 1)
    incy = stride(y, 1)
    BLAS.rot!(n, x, incx, y, incy, c, s)

    queue = global_queue(context(d_x), device(d_x))
    incx = stride(d_x, 1)
    incy = stride(d_y, 1)
    @assert length(d_x) >= 1 + (n - 1)*abs(incx)
    @assert length(d_y) >= 1 + (n - 1)*abs(incy)

    (T == Float32)    && oneMKL.onemklSrot(sycl_queue(queue), n, d_x, incx, d_y, incy, c, s)
    (T == Float64)    && oneMKL.onemklDrot(sycl_queue(queue), n, d_x, incx, d_y, incy, c, s)
    (T == ComplexF32) && oneMKL.onemklCrot(sycl_queue(queue), n, d_x, incx, d_y, incy, c, s)
    (T == ComplexF64) && oneMKL.onemklZrot(sycl_queue(queue), n, d_x, incx, d_y, incy, c, s)
    
    @show x
    @show Array(d_x)
    @show c
    @show s
    @assert x == Array(d_x)
end

main()
