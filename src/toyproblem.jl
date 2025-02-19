function toyproblem0(D; seed = 1)

    rng = MersenneTwister(seed)

    A = randn(rng, D, D); A = Diagonal(A'A)

    d = MvNormal(zeros(D), A)

    return x -> logpdf(d, x), d

end


function toyproblem1(D; seed = 1)

    rng = MersenneTwister(seed)

    A = randn(rng, D, D); A = A'A

    d = MvNormal(zeros(D), A)

    return x -> logpdf(d, x), d

end


function toyproblem2()

    w(x) = sin(2π*x/2)

    U(z) = (z[2]-w(z[1]))^2

    loglikel(z) = -U(z)

    # d = MvNormal(zeros(2), 1.0)

    logprior(z) = -0.5*mapreduce(abs2, +, z) # Distributions.logpdf(d, z)

    logp(z) = loglikel(z) + logprior(z)

    return logp
    
end