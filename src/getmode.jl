function getmode(logp, x; gradlogp = nothing, opt = Optim.Options(show_trace = true, show_every = 10, iterations = 1_000_000))

    helper(x) = -logp(x)

    _getmode(helper, x, gradlogp, opt)

end

_getmode(helper, x, ::Nothing, opt) = optimize(helper, x, NelderMead(), opt).minimizer

function _getmode(helper, x, gradlogp, opt)

    gradhelper!(s, p) = copyto!(s, -gradlogp(p))

    optimize(helper, gradhelper!, x, LBFGS(), opt).minimizer

end



function getmode_and_eigenvectors(logp, x; gradlogp = nothing, opt = Optim.Options(show_trace = true, show_every = 10, iterations = 1_000_000))

    m = getmode(logp, x; gradlogp = gradlogp, opt = opt)

    V = geteigenvectors(logp, m)

    return m, V

end