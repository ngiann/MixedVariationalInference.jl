optimise(elbo::ElboMVI, opt) = _optimise(elbo, [zeros(elbo.D);ones(elbo.D)], opt, Val(hasgradient(elbo)))

optimise(elbo::ElboMVI, res::Optim.OptimizationResults, opt) = _optimise(elbo, getsolution(res), opt, Val(hasgradient(elbo)))

optimise(elbo::ElboMVI, x₀, opt) = _optimise(elbo, x₀, opt, Val(hasgradient(elbo)))


function _optimise(elbo::ElboMVI, x₀::Vector, opt, ::Val{false})

    @assert(length(x₀) == numparam(elbo))

    helper(x) = -elbo(x)

    Optim.optimize(helper, x₀, NelderMead(), opt)

end

function _optimise(elbo::ElboMVI, x₀::Vector, opt, ::Val{true})

    @assert(length(x₀) == numparam(elbo))

    helper(x) = -elbo(x)

    gradhelper!(s, p) = copyto!(s, -grad(elbo, p))

    Optim.optimize(helper, gradhelper!, x₀, ConjugateGradient(), opt)

end