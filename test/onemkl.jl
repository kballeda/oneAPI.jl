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
    @testset for elty in [Float32]
        alpha = rand(elty)
        beta = rand(elty)

        @testset "gemv" begin
            @test testf(*, rand(elty, m, n), rand(elty, n))
            @test testf(*, transpose(rand(elty, m, n)), rand(elty, m))
            @test testf(*, rand(elty, m, n)', rand(elty, m))
            x = rand(elty, m)
            A = rand(elty, m, m + 1 )
            y = rand(elty, m)
            dx = oneArray(x)
            dA = oneArray(A)
            dy = oneArray(y)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
            A = rand(elty, m + 1, m )
            dA = oneArray(A)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
            x = rand(elty, m)
            A = rand(elty, n, m)
            dx = oneArray(x)
            dA = oneArray(A)
            alpha = rand(elty)
            dy = oneMKL.gemv('N', alpha, dA, dx)
            hy = collect(dy)
            @test hy ≈ alpha * A * x
            dy = oneMKL.gemv('N', dA, dx)
            hy = collect(dy)
            @test hy ≈ A * x
        end
    end
end
