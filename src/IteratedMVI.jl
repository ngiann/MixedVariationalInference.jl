module IteratedMVI

    using Distributions
    using ForwardDiff
    using LinearAlgebra
    using OnlineStats
    using Optim
    using Printf
    using Random
    using StatsFuns

    include("diagnostic_logp_gradlogp.jl")
    include("ElboMVI.jl")
    include("entropy.jl")
    include("generatelatentZ.jl")
    include("geteigenvectors.jl")
    include("getmode.jl")
    include("numerical_KLD.jl")
    include("optimise.jl")
    include("toyproblem.jl")
    include("unpack.jl")
    include("util.jl")


    export diagnostic_logp_gradlogp, elbofy_mvi, getmode_and_eigenvectors, getsolution, grad, numerical_KLD, numparam, optimise, posterior, testelbo, unpack 
    export toyproblem0, toyproblem1, toyproblem2

end
