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
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        
        @testset "trmm!" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*A*B
            oneMKL.trmm!('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            h_C = Array(dB)
            @test C ≈ h_C
        end
        @testset "trmm" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*A*B
            oneMKL.trmm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            h_C = Array(dB)
            @test C ≈ h_C
        end

        @testset "left trsm!" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(A\B)
            dC = copy(dB)
            oneMKL.trsm!('L','U','N','N',alpha,dA,dC)
            @test C ≈ Array(dC)
        end

        @testset "left trsm" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(A\B)
            dC = oneMKL.trsm('L','U','N','N',alpha,dA,dB)
            @test C ≈ Array(dC)
        end

        @testset "left trsm (adjoint)" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(adjoint(A)\B)
            dC = oneMKL.trsm('L','U','C','N',alpha,dA,dB)
            @test C ≈ Array(dC)
        end

        @testset "left trsm (transpose)" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(transpose(A)\B)
            dC = oneMKL.trsm('L','U','T','N',alpha,dA,dB)
            @test C ≈ Array(dC)
        end

        let A = rand(T, m,m), B = triu(rand(T, m, m)), alpha = rand(T)
            dA = oneArray(A)
            dB = oneArray(B)

            @testset "right trsm!" begin
                C = alpha*(A/B)
                dC = copy(dA)
                oneMKL.trsm!('R','U','N','N',alpha,dB,dC)
                @test C ≈ Array(dC)
            end

            @testset "right trsm" begin
                C = alpha*(A/B)
                dC = oneMKL.trsm('R','U','N','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
            @testset "right trsm (adjoint)" begin
                C = alpha*(A/adjoint(B))
                dC = oneMKL.trsm('R','U','C','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
            @testset "right trsm (transpose)" begin
                C = alpha*(A/transpose(B))
                dC = oneMKL.trsm('R','U','T','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
        end
    end
end