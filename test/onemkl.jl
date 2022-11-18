using oneAPI
using oneAPI.oneMKL

using LinearAlgebra

m = 20
n = 35
k = 13

############################################################################################
@testset "level 1" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        @testset "copy" begin
            A = oneArray(rand(T, m))
            B = oneArray{T}(undef, m)
            oneMKL.copy!(m,A,B)
            @test Array(A) == Array(B)
        end

        @testset "nrm2" begin
            @test testf(norm, rand(T,m))
        end

        @testset "iamax/iamin" begin
            a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
            ca = oneArray(a)
            @test BLAS.iamax(a) == oneMKL.iamax(ca)
            @test oneMKL.iamin(ca) == 3
        end

        @testset "swap" begin
            x = rand(T, m)
            y = rand(T, m)
            dx = oneArray(x)
            dy = oneArray(y)
            oneMKL.swap!(m, dx, dy)
            @test Array(dx) == y
            @test Array(dy) == x
        end
    end
end

@testset "level 2" begin
    @testset for T in intersect(eltypes, [ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        B = rand(T,m,n)
        C = rand(T,m,n)
        d_B = oneArray(B)
        d_C = oneArray(C)
        hA = rand(T,m,m)
        hA = hA + hA'
        dhA = oneArray(hA)
        @testset "hemm!" begin
            # compute
            C = alpha*(hA*B) + beta*C
            oneMKL.hemm!('L','L',alpha,dhA,d_B,beta,d_C)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end

        @testset "hemm" begin
            C = hA*B
            d_C = oneMKL.hemm('L','U',dhA,d_B)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end

    end
end

