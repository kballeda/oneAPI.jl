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

# level 2 tests
@testset "level 2" begin
    @testset for T in intersect(eltypes, [Float32, Float64])
        @testset "triangular" begin
             # generate triangular matrix
             A = rand(T,m,m)
             # restrict to 3 bands
             nbands = 3
             @test m >= 1+nbands
             A = bandex(A,0,nbands)
             # convert to 'upper' banded storage format
             AB = band(A,0,nbands)
             d_AB = oneArray(AB)
             x = rand(T,m)
             d_x = oneArray(x)
             @testset "tbmv!" begin
                y = rand(T, m)
                # move to host
                d_y = oneArray(y)
                # tbmv!
                oneMKL.tbmv!('U','N','N',nbands,d_AB,d_y)
                y = A*y
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "tbmv" begin
                # tbmv
                d_y = oneMKL.tbmv('U','N','N',nbands,d_AB,d_x)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

        end
    end
end
