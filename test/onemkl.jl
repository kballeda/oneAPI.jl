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
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        @testset "triangular" begin
             @testset "tbmv!" begin
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
                # tbmv
                d_y = oneMKL.tbmv('U','N','N',nbands,d_AB,d_x)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "tbsv!" begin
                # generate triangular matrix
                A = rand(T,m,m)
                # restrict to 3 bands
                nbands = 3
                #@test m >= 1+nbands
                A = bandex(A,0,nbands)
                # convert to 'upper' banded storage format
                AB = band(A,0,nbands)
                d_AB = oneArray(AB)
                x = rand(T,m)
                d_x = oneArray(x)
                d_y = copy(d_x)
                #tbsv!
                oneMKL.tbsv!('U','N','N',nbands,d_AB,d_y)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "tbsv" begin
                # generate triangular matrix
                A = rand(T,m,m)
                # restrict to 3 bands
                nbands = 3
                #@test m >= 1+nbands
                A = bandex(A,0,nbands)
                # convert to 'upper' banded storage format
                AB = band(A,0,nbands)
                d_AB = oneArray(AB)
                x = rand(T,m)
                d_x = oneArray(x)
                d_y = oneMKL.tbsv('U','N','N',nbands,d_AB,d_x)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            
            @testset "trmv!" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A)
                x = rand(T, m)
                dx = oneArray(x)
                d_y = copy(dx)
                # execute trmv!
                oneMKL.trmv!('U','N','N',dA,d_y)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
                #@test_throws DimensionMismatch oneMKL.trmv!('U','N','N',dA, rand(T,m+1))
            end

            @testset "trmv" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A) 
                x = rand(T, m)
                dx = oneArray(x)
                d_y = copy(dx)
                d_y = oneMKL.trmv('U','N','N',dA,dx)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trsv!" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A) 
                x = rand(T, m)
                dx = oneArray(x)
                d_y = copy(dx)
                # execute trsv!
                oneMKL.trsv!('U','N','N',dA,d_y)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
                #@test_throws DimensionMismatch CUBLAS.trsv!('U','N','N',dA,CUDA.rand(elty,m+1))
            end
    
            @testset "trsv" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A) 
                x = rand(T, m)
                dx = oneArray(x)
                d_y = oneMKL.trsv('U','N','N',dA,dx)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trsv (adjoint)" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A) 
                x = rand(T, m)
                dx = oneArray(x)
                d_y = oneMKL.trsv('U','C','N',dA,dx)
                y = adjoint(A)\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trsv (transpose)" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A) 
                x = rand(T, m)
                dx = oneArray(x)
                d_y = oneMKL.trsv('U','T','N',dA,dx)
                y = transpose(A)\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

        end
    end
end
