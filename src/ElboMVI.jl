struct ElboMVI{T1, T2, F1, F2}
    Z::T1
    D::Int64
    S::Int64
    C::Matrix{T2} # pre-allocated to hold root of cov matrix
    logp::F1
    gradlogp::F2
    m::Vector{T2} # holds mode
    V::Matrix{T2} # holds eigenvectors
end

#----------------------------------------------------------------------------------------------------------------
function Base.show(io::IO, ::MIME"text/plain", elbo::ElboMVI)
#----------------------------------------------------------------------------------------------------------------

    println(io, "Deterministic mixed variational inference (D = ", elbo.D, ", S = ", elbo.S, ").")
    
    println(io, "Number of variational parameters is ", numparam(elbo), ".")

    hasgradient(elbo) ? println(io, "Gradient for logp has been provided.") : println(io, "No gradient for logp provided.")

    if Threads.nthreads() > 1 
        print(io, "There are ", Threads.nthreads(), " available.")
    else
        print(io, "There is ", Threads.nthreads(), " available.")
    end
    
end

#----------------------------------------------------------------------------------------------------------------
function elbofy_mvi(logp, D, S; gradlogp = nothing, rng::AbstractRNG = Xoshiro(1))
#----------------------------------------------------------------------------------------------------------------
    
    m = zeros(D)
    
    V = 1.0 * Matrix(I, D, D)

    elbofy_mvi(logp, m, V, S; gradlogp = gradlogp, rng = rng)
    
end


#----------------------------------------------------------------------------------------------------------------
function elbofy_mvi(logp, m::Vector{T}, V::Matrix{T}, S₀; gradlogp = nothing, rng::AbstractRNG = Xoshiro(1)) where T<:Real
#----------------------------------------------------------------------------------------------------------------

    S = roundup_S(S₀)

    D = size(V, 1); @assert(size(V, 2) == D)

    Z = generatelatentZ(; S = S, D = D, rng = rng)

    C = zeros(D, D) # preallocated covariance root

    return ElboMVI(Z, D, S, C, logp, gradlogp, m, V)

end


(elbo::ElboMVI)(res::Optim.OptimizationResults) = elbo(getsolution(res))

(elbo::ElboMVI)(param) = elbo(param, elbo.Z)


    

#----------------------------------------------------------------------------------------------------------------
function (elbo::ElboMVI)(param::Vector{T}, Z)::T where T<:Real
#----------------------------------------------------------------------------------------------------------------

    S = length(Z)

    # Retrieve variational parameters
    μ::Vector{T}, Esqrt::Vector{T} = unpack(elbo, param)

    # instantiate matrix covariance root and store it in elbo.C
    getcovroot!(elbo, Esqrt) 
    
    # Chuck samples - no allocations
    Zchunks = chunckZ(Z)
    
    # Each thread computes part of the monte carlo estimate of expected log-likelihood
    tasks = map(Zchunks) do Zchunk

        Threads.@spawn begin

            local Elogl = zero(T)

            local θ = zeros(T, elbo.D) # pre-allocate pre-thread

            for z in Zchunk

                mul!(θ, elbo.C, z)

                θ .+= μ

                Elogl += elbo.logp(θ)
                
            end

            Elogl

        end
        
    end

    # accummulate results from threads
    Elogl = reduce(+, fetch.(tasks)) / S

    # Calculate exact Gaussian entropy
    ℋ = entropy(Esqrt)
    
    # Approximate lower bound is sum of both terms
    return Elogl + ℋ
        
end



#----------------------------------------------------------------------------------------------------------------
function grad(elbo::ElboMVI, param::Vector{T}) where T<:Real
#----------------------------------------------------------------------------------------------------------------
        
    S = length(elbo.Z)

    μ::Vector{T}, Esqrt::Vector{T} = unpack(elbo, param)

    # instantiate matrix covariance root and store it in elbo.C
    getcovroot!(elbo, Esqrt) 

    V, D = elbo.V, elbo.D # convenient names

    # Chuck samples - no allocations
    Zchunks = chunckZ(elbo.Z)

    # Each thread computes part of the monte carlo estimate of expected log-likelihood
    tasks = map(Zchunks) do Zchunk

        Threads.@spawn begin

            local gradient = zeros(T, 2D) # allocate new vector per thread to store gradient
            
            local θ = zeros(T, D) # per thread pre-allocated vector to instantiated parameters

            local g = zeros(T, D) # per thread pre-allocated vector to store gradient

            local buffer = zeros(T, D) # per thread pre-allocated vector to store multiplication result

            for z in Zchunk

                mul!(θ, elbo.C, z)

                θ .+= μ

                copyto!(g, elbo.gradlogp(θ))

                # gradient wrt μ
                for i in 1:D
                    @inbounds gradient[i] += g[i]
                end

                # gradient wrt Esqrt
                mul!(buffer, V', g)

                buffer .*= z

                for i in 1:D
                    @inbounds gradient[D+i] += buffer[i]
                end

            end

            # return gradient of μ and Esqrt computed by thread
            gradient

        end

    end

    # allocate array to store returned gradient
    finalgradient = mapreduce(fetch, +, tasks)

    # divide by number of samples to form approximate expectation
    finalgradient ./= S
    
    # entropy contribution to covariance
    finalgradient[D+1:2D] .+= 1.0./Esqrt
    
    # return gradient of μ and Esqrt
    return finalgradient

end

testelbo(elbo, param; Stest = elbo.S * 10, rng = Xoshiro(0)) = elbo(param, IteratedMVI.generatelatentZ(S = Stest, D = elbo.D, rng = rng))

hasgradient(elbo::ElboMVI) = true

hasgradient(elbo::ElboMVI{T1, T2, F1, F2}) where {T1<:Any, T2<:Any, F1<:Any, F2<:Nothing} = false