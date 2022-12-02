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

@testset "level 3" begin
    @testset for T in intersect(eltypes, [ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        
        @testset "hemm!" begin
            B = rand(T,m,n)
            C = rand(T,m,n)
            d_B = oneArray(B)
            d_C = oneArray(C)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            # compute
            C = alpha*(hA*B) + beta*C
            oneMKL.hemm!('L','L',alpha,dhA,d_B,beta,d_C)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end

        @testset "hemm" begin
            B = rand(T,m,n)
            C = rand(T,m,n)
            d_B = oneArray(B)
            d_C = oneArray(C)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            
            C = hA*B
            d_C = oneMKL.hemm('L','U',dhA,d_B)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end

        @testset "herk!" begin
            B = rand(T,m,n)
            C = rand(T,m,n)
            d_B = oneArray(B)
            d_C = oneArray(C)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            A = rand(T,m,k)
            d_A = oneArray(A)
            d_C = oneArray(dhA)
            oneMKL.herk!('U','N',real(alpha),d_A,real(beta),d_C)
            C = real(alpha)*(A*A') + real(beta)*hA
            C = triu(C)
            # move to host and compare
            h_C = Array(d_C)
            h_C = triu(C)
            @test C ≈ h_C
        end

        @testset "herk" begin
            B = rand(T,m,n)
            C = rand(T,m,n)
            d_B = oneArray(B)
            d_C = oneArray(C)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            A = rand(T,m,k)
            d_A = oneArray(A)
            d_C = oneMKL.herk('U','N',d_A)
            C = A*A'
            C = triu(C)
            # move to host and compare
            h_C = Array(d_C)
            h_C = triu(C)
            @test C ≈ h_C
        end

        @testset "her2k!" begin
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
            elty1 = T
            elty2 = real(T)
            # generate parameters
            α = rand(elty1)
            β = rand(elty2)
            C = C + C'
            d_C = oneArray(C)
            C = α*(A*B') + conj(α)*(B*A') + β*C
            oneMKL.her2k!('U','N',α,d_A,d_B,β,d_C)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch oneMKL.her2k!('U','N',α,d_A,d_Bbad,β,d_C)
        end

        @testset "her2k" begin
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
            C = A*B' + B*A'
            d_C = oneMKL.her2k('U','N',d_A,d_B)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
        end
    end
end

