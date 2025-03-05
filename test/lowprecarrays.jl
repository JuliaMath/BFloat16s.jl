@testset "lowprecarrays" begin
    A = rand(Float32, 10,10)
    lpA = LowPrecArray(A)

    @test lpA isa LowPrecArray
    @test size(A) == size(lpA)

   lpA[3] = 4
    @test lpA[3] == A[3]

    a = rand(Float32)
    ab = BFloat16s.ExpandingBFloat16(a)
    b = rand(Float32)
    bb = BFloat16s.ExpandingBFloat16(b)

    @test ab isa BFloat16s.ExpandingBFloat16
    @test ab.a isa BFloat16
    @test ab * bb isa Float32

    B = LowPrecArray(rand(Float32, 10,10))
    C = LowPrecArray(rand(Float32, 10,10))

    lpA .= B*C

end # @testset "lowprecarrays"