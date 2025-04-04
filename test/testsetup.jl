module TensorKitTensorsTestSetup

export test_operator, operator_sum, get_lowest_eigenvalues

using Test
using TensorKit
using TensorKit: scalartype
using LinearAlgebra: eigvals

function operator_sum(O::AbstractTensorMap; L::Int=4)
    I = id(space(O, 1))
    n = numin(O)
    return sum(1:(L - n + 1)) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - n)), i, O))
    end
end

function test_operator(O1::AbstractTensorMap, O2::AbstractTensorMap; L::Int=4,
                       isapproxkwargs...)
    H1 = operator_sum(O1; L)
    H2 = operator_sum(O2; L)
    eigenvals1 = mapreduce(vcat, eigvals(H1)) do (c, vals)
        return repeat(vals, dim(c))
    end
    eigenvals2 = mapreduce(vcat, eigvals(H2)) do (c, vals)
        return repeat(vals, dim(c))
    end
    @test isapprox(sort!(eigenvals1; by=real), sort!(eigenvals2; by=real);
                   isapproxkwargs...)
end

function get_lowest_eigenvalues(O1::AbstractTensorMap, n::Int; L::Int=4)
    H = operator_sum(O1; L)
    eigenvals = mapreduce(vcat, eigvals(H)) do (c, vals)
        return repeat(vals, dim(c))
    end
    sort!(eigenvals; by=real)
    if n == -1
        return eigenvals
    end
    return eigenvals[1:n]
end

end
