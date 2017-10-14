module joshnet

export truncated_gaussian_init
export Tensor

function truncated_gaussian_init(shape::Tuple{<:Integer, <:Integer};
                                 dtype::DataType=Float64)
    assert(length(shape) == 2)
    if dtype != Float64
        return convert(Array{dtype, 2}, rand(shape...))
    else
        return rand(shape...)
    end
end

mutable struct Tensor{T<:AbstractFloat}
    weights::Matrix{T}
    shape::Tuple{<:Integer, <:Integer}
end
Tensor(w::Matrix{T}) where {T<:AbstractFloat} = Tensor(w, size(w))
Tensor(height::Integer, width::Integer; init::Function=truncated_gaussian_init,
       dtype::DataType=Float64) = Tensor{dtype}(init((height, width)),
                                                (height, width))

mutable struct Variable{T<:AbstractFloat}
    tensor::Tensor
    gradients::Matrix{T}
end
Variable(t::Tensor; dtype::DataType=Float64) = Variable(t, zero(Matrix{dtype}(size(t.weights))))

# Functions on tensors
function reduce_mean(v::Variable; axis::Vector{<:Integer}=Vector{Int32}())
    if length(axis) == 0
        v.gradients += 1.0 / reduce(*, v.tensor.shape)
        return mean(v.tensor.weights)
    end
end
end
