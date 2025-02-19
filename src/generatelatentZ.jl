function generatelatentZ(; S = S, D = D, rng=rng)

    Z  = [randn(rng, D) for _ in 1:S]

    return Z

end
