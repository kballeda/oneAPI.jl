using oneAPI
using oneAPI.oneMKL: band, bandex

using LinearAlgebra

m = 20

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
    @testset for T in intersect(eltypes, [Float32, Float64])
        alpha = rand(T)
        beta = rand(T)
        A = rand(T,m,m)
        A = A + A'
        nbands = 3
        @test m >= 1+nbands
        A = bandex(A,nbands,nbands)
        # convert to 'upper' banded storage format
        AB = band(A,0,nbands)
        # construct x
        x = rand(T,m)
        d_AB = oneArray(AB)
        d_x = oneArray(x)
        @testset "sbmv!" begin
            y = rand(T,m)
            d_y = oneArray(y)
            # sbmv!
            oneMKL.sbmv!('U',nbands,alpha,d_AB,d_x,beta,d_y)
            y = alpha*(A*x) + beta*y
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end
        @testset "sbmv" begin
            d_y = oneMKL.sbmv('U',nbands,d_AB,d_x)
            y = A*x
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end
    end
end