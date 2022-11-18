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

@testset "level 3" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        B = rand(T,m,n)
        C = rand(T,m,n)
        Bbad = rand(T,m+1,n+1)
        d_B = oneArray(B)
        d_C = oneArray(C)
        d_Bbad = oneArray(Bbad)
        sA = rand(T,m,m)
        sA = sA + transpose(sA)
        dsA = oneArray(sA)

        @testset "symm!" begin
            oneMKL.symm!('L','U',alpha,dsA,d_B,beta,d_C)
            C = (alpha*sA)*B + beta*C
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch oneMKL.symm!('L','U',alpha,dsA,d_Bbad,beta,d_C)
        end

        @testset "symm" begin
            d_C = oneMKL.symm('L','U',dsA,d_B)
            C = sA*B
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch oneMKL.symm('L','U',dsA,d_Bbad)
        end

        @testset "syrk" begin
            A = rand(T,m,k)
            d_A = oneArray(A)
            d_C = oneMKL.syrk('U','N',d_A)
            C = A*transpose(A)
            C = triu(C)
            # move to host and compare
            h_C = Array(d_C)
            h_C = triu(C)
            @test C ≈ h_C
        end

        A = rand(T,m,k)
        B = rand(T,m,k)
        Bbad = rand(T,m+1,k+1)
        C = rand(T,m,m)
        C = C + transpose(C)
        # move to device
        d_A = oneArray(A)
        d_B = oneArray(B)
        d_Bbad = oneArray(Bbad)
        d_C = oneArray(C)
        @testset "syr2k!" begin
            # compute
            C = alpha*(A*transpose(B) + B*transpose(A)) + beta*C
            oneMKL.syr2k!('U','N',alpha,d_A,d_B,beta,d_C)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch oneMKL.syr2k!('U','N',alpha,d_A,d_Bbad,beta,d_C)
        end

        @testset "syr2k" begin
            C = alpha*(A*transpose(B) + B*transpose(A))
            d_C = oneMKL.syr2k('U','N',alpha,d_A,d_B)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
        end

    end
end