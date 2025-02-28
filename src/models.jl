"""
AbstractMLmodel
Supertype of both ML models used
"""
abstract type AbstractMLmodel{T} end

"""
AbstractESNmodel{T} <: AbstractMLmodel{T}
Supertype of all ESN models
"""
abstract type AbstractESNmodel{T} <: AbstractMLmodel{T} end 

"""
AbstractNDEmodel{T} <: AbstractMLmodel{T}
Supertype of all NDE models
"""
abstract type AbstractNDEmodel{T} <: AbstractMLmodel{T} end 


"""@with_kw struct ESN{T} <: AbstractESNmodel
    
end"""