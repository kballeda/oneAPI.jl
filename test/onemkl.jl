using oneAPI
using oneAPI.oneMKL: band, bandex

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
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        @testset "banded methods" begin
            # bands
            ku = 2
            kl = 3

            # generate banded matrix
            A = rand(T, m,n)
            A = bandex(A, kl, ku)

            # get packed format
            Ab = band(A, kl, ku)
            d_Ab = oneArray(Ab)
            x = rand(T, n)
            d_x = oneArray(x)

            @testset "gbmv!" begin
                # Test: y = alpha * A * x + beta * y
                y = rand(T, m)
                d_y = oneArray(y)
                oneMKL.gbmv!('N', m, kl, ku, alpha, d_Ab, d_x, beta, d_y)
                BLAS.gbmv!('N', m, kl, ku, alpha, Ab, x, beta, y)
                h_y = Array(d_y)
                @test y ≈ h_y

                # Test: y = alpha * transpose(A) * x + beta * y
                x = rand(T, n)
                d_x = oneArray(x)
                y = rand(T,m)
                d_y = oneArray(y)
                oneMKL.gbmv!('T', m, kl, ku, alpha, d_Ab, d_y, beta, d_x)
                BLAS.gbmv!('T', m, kl, ku, alpha, Ab, y, beta, x)
                h_x = Array(d_x)
                @test x ≈ h_x

                # Test: y = alpha * A'*x + beta * y
                x = rand(T,n)
                d_x = oneArray(x)
                y = rand(T,m)
                d_y = oneArray(y)
                oneMKL.gbmv!('C', m, kl, ku, alpha, d_Ab, d_y, beta, d_x)
                BLAS.gbmv!('C', m, kl, ku, alpha, Ab, y, beta, x)
                h_x = Array(d_x)
                @test x ≈ h_x

                # Test: alpha=1 version without y
                d_y = oneMKL.gbmv('N', m, kl, ku, d_Ab, d_x)
                y = BLAS.gbmv('N', m, kl, ku, Ab, x)
                h_y = Array(d_y)
                @test y ≈ h_y

            end
        end
    end
end

