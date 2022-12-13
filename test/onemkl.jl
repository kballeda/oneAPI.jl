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

        @testset "axpy" begin
            alpha = rand(T,1)
            @test testf(axpy!, alpha[1], rand(T,m), rand(T,m))
        end
        
        @testset "scal" begin
            # Test scal primitive [alpha/x: F32, F64, CF32, CF64]
            alpha = rand(T,1)
            @test testf(rmul!, rand(T,m), alpha[1])

            # Test scal primitive [alpha - F32, F64, x - CF32, CF64] 
            A = rand(T,m)
            gpuA = oneArray(A)
            if T === ComplexF32
                alphaf32 = rand(Float32, 1)
                oneMKL.scal!(m, alphaf32[1], gpuA)
                @test Array(A .* alphaf32[1]) ≈ Array(gpuA)
            end

            if T === ComplexF64
                alphaf64 = rand(Float64, 1)
                oneMKL.scal!(m, alphaf64[1], gpuA)
                @test Array(A .* alphaf64[1]) ≈ Array(gpuA)
            end	    
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

        @testset "asum" begin
            @test testf(BLAS.asum, rand(T,m))
        end
    end
end

@testset "level 2" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        # generate matrices
        bA = [rand(T,m,k) for i in 1:10]
        bB = [rand(T,k,n) for i in 1:10]
        bC = [rand(T,m,n) for i in 1:10]
        # copy data to device
        bd_A = oneArray{T, 2}[]
        bd_B = oneArray{T, 2}[]
        bd_C = oneArray{T, 2}[]
        bd_bad = oneArray{T, 2}[]
        for i in 1:length(bA)
            push!(bd_A, oneArray(bA[i]))
            push!(bd_B, oneArray(bB[i]))
            push!(bd_C, oneArray(bC[i]))
            if i < length(bA) - 2
                push!(bd_bad, oneArray(bC[i]))
            end
        end

        @testset "gemm_batched!" begin
            # C = (alpha*A)*B + beta*C
            oneMKL.gemm_batched!('N','N',alpha,bd_A,bd_B,beta,bd_C)
            for i in 1:length(bd_C)
                bC[i] = (alpha*bA[i])*bB[i] + beta*bC[i]
                h_C = Array(bd_C[i])
                #compare
                @test bC[i] ≈ h_C
            end
            @test_throws DimensionMismatch oneMKL.gemm_batched!('N','N',alpha,bd_A,bd_bad,beta,bd_C)
        end

        @testset "gemm_batched" begin
            bd_C = oneMKL.gemm_batched('N','N',bd_A,bd_B)
            for i in 1:length(bA)
                bC = bA[i]*bB[i]
                h_C = Array(bd_C[i])
                @test bC ≈ h_C
            end
            @test_throws DimensionMismatch oneMKL.gemm_batched('N','N',alpha,bd_A,bd_bad)
        end
    end
end
