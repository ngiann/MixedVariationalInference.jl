function entropy(Esqrt::Vector{T}) where T<:Real

    D = length(Esqrt)

    ℋ = zero(T) + 0.5*log(2*π*ℯ) * D

    for i in eachindex(Esqrt)
        
        ℋ += 0.5*log(abs2(Esqrt[i]))

    end 
    
    return ℋ

end