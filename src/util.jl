numparam(elbo::ElboMVI) = elbo.D * 2

getcovroot(elbo, Esqrt) = elbo.V * Diagonal(Esqrt)

getcovroot!(elbo, Esqrt) = mul!(elbo.C, elbo.V, Diagonal(Esqrt))

function getμΣ(elbo::ElboMVI, param)

    μ, Esqrt = unpack(elbo, param)

    Σ = elbo.V * Diagonal(Esqrt.^2 ) * elbo.V' 

    return μ, Symmetric(Σ)
    
end

getsolution(res::Vector) = res

getsolution(res::Optim.OptimizationResults) = Optim.minimizer(res)   


posterior(elbo::ElboMVI, res::Optim.OptimizationResults) = posterior(elbo, getsolution(res))

posterior(elbo::ElboMVI, param::Vector) = MvNormal(getμΣ(elbo, param)...)



sizeofchuncks(S) = Int(S / Threads.nthreads())

chunckZ(Z) = Iterators.partition(Z, sizeofchuncks(length(Z)))



is_valid_S(S) = rem(S, Threads.nthreads()) == 0

roundup_S(S) = S + Threads.nthreads() - rem(S, Threads.nthreads())



