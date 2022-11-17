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

        @testset "symv tests" begin
            x = rand(T,m)
            sA = rand(T, m, m)
            sA = sA + transpose(sA)
            dsA = oneArray(sA)
            dx = oneArray(x)
            @testset "symv!" begin
                # generate vectors
                y = rand(T,m)
                # copy to device
                dy = oneArray(y)
                # execute on host
                BLAS.symv!('U',alpha,sA,x,beta,y)
                # execute on device
                oneMKL.symv!('U',alpha,dsA,dx,beta,dy)
                # compare results
                hy = Array(dy)
                @test y ≈ hy
            end

            @testset "symv" begin
                y = BLAS.symv('U',sA,x)
                # execute on device
                dy = oneMKL.symv('U',dsA,dx)
                # compare results
                hy = Array(dy)
                @test y ≈ hy
            end
        end

        @testset "syr!" begin
            x = rand(T,m)
            sA = rand(T, m, m)
            sA = sA + transpose(sA)
            dsA = oneArray(sA)
            dx = oneArray(x)
            dB = copy(dsA)
            oneMKL.syr!('U',alpha,dx,dB)
            B = (alpha*x)*transpose(x) + sA
            # move to host and compare upper triangles
            hB = Array(dB)
            B = triu(B)
            hB = triu(hB)
            @test B ≈ hB
        end
    end
end
