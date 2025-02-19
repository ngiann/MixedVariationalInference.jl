function diagnostic_logp_gradlogp(np)

    # return mock non-allocating, stable functions for testing code
    
    O = zeros(Float64, np)
    
    logp = (_) -> 0.0
    gradlogp = (_) -> O

    return logp, gradlogp

end