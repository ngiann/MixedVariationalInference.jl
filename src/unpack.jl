function unpack(elbo::ElboMVI, param)

    @assert(length(param) == numparam(elbo))

    D = elbo.D

    MARK = 0

    μ = elbo.m .+ param[MARK+1:MARK+D]; MARK += D
    
    Esqrt = param[MARK+1:MARK+D]; MARK += D

    @assert(length(param) == MARK) # ensure all parameters are used up

    return μ, Esqrt

end