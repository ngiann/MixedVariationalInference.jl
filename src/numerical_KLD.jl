function numerical_KLD(d, logp; numsamples = 10_000, rng::AbstractRNG = Xoshiro(1))

    stats = Series(Mean(), Variance())

    for _ in 1:numsamples

        sample = Distributions.rand(rng, d)

        aux = Distributions.logpdf(d, sample) - logp(sample)

        OnlineStats.fit!(stats, aux)

    end

    μ, v = OnlineStats.value.(stats)

    return μ, sqrt(v)/sqrt(numsamples)
end
