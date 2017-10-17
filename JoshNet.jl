module JoshNet

export Tensor, None
export +, *, -, ^, abs, exp, relu, reduce_mean, reduce_sum, softmax
export emptytensor, gaussian_tensor, variance_scaled_tensor
export SGDOptimizer, SGDWithMomentum, optimize!
export fc_layer

import Base.*
import Base.+
import Base.-
import Base.abs
import Base.exp
import Base.^

abstract type JNNode end
abstract type Op <: JNNode end
abstract type BinaryOp <: Op end
abstract type UnaryOp <: Op end

struct None <: JNNode
end
function backward(node::None)
end

# ------------------------- DATA TYPES ----------------------------------------

mutable struct Tensor <: JNNode
    data::Matrix{Float32}
    grad::Matrix{Float32}
    parent::JNNode
    train::Bool
end
Tensor() = Tensor(Matrix{Float32}(0, 0), Matrix{Float32}(0, 0), None(), false)
function Tensor(m::Matrix{Float32}; trainable::Bool=true)
    shape = (size(m)[1], size(m)[2])
    return Tensor(m, zeros(Matrix{Float32}(shape)), None(), trainable)
end
function emptytensor(cols::Integer, rows::Integer; trainable::Bool=true)
    return Tensor(zeros{Float32}(cols, rows),
                  zeros{Float32}(cols, rows),
                  None(),
                  trainable)
end
function apply_gradients!(t::Tensor)
    if t.train
        t.data -= t.grad
    end
end
function zero_gradients!(t::Tensor)
    t.grad = zeros(t.grad)
end

# ------------------------- INIT ---------------------------------------------

function gaussian_tensor(height::Integer, width::Integer; mean::Number=0,
                         variance::Number=0.1, trainable::Bool=true)
    return Tensor((mean + randn(Float32, height, width)) / variance,
                  trainable=trainable)
end

function variance_scaled_tensor(height::Integer, width::Integer; trainable::Bool=true)
    return Tensor(randn(Float32, height, width) / (10 * height),
                  trainable=trainable)
end

# ------------------------- OPS ----------------------------------------------

# ==== REDUCE MEAN ==== #

struct ReduceMeanOp <: UnaryOp
    parent::JNNode
    child::JNNode
    dims::Vector{Integer}
end
function forward(op::ReduceMeanOp)
    op.child.parent = op
    result = op.parent.data
    for dim in op.dims
        result = mean(result, dim)
    end
    op.child.data = result
    op.child.grad = zeros(Array{Float32}(size(result)...))
    return op.child
end
function backward(op::ReduceMeanOp)
    grad = zeros(Matrix{Float32}(size(op.parent.grad)...))
    divisor = 1.0
    for dim in op.dims
        divisor *= size(op.parent.data)[dim]
    end
    grad .+= (op.child.grad ./ divisor)
    return grad
end
function reduce_mean(t1::Tensor; axis::Vector{<:Integer}=Integer[1, 2])
    r = ReduceMeanOp(t1, Tensor(), axis)
    return forward(r)
end

# ==== REDUCE SUM ==== #

struct ReduceSumOp <: UnaryOp
    parent::JNNode
    child::JNNode
    dims::Vector{Integer}
end
function forward(op::ReduceSumOp)
    op.child.parent = op
    result = op.parent.data
    for dim in op.dims
        result = sum(result, dim)
    end
    op.child.data = result
    op.child.grad = zeros(Array{Float32}(size(result)...))
    return op.child
end
function backward(op::ReduceSumOp)
    grad = zeros(Matrix{Float32}(size(op.parent.grad)...))
    grad .+= op.child.grad
    return grad
end
function reduce_sum(t1::Tensor; axis::Vector{<:Integer}=Integer[1, 2])
    r = ReduceSumOp(t1, Tensor(), axis)
    return forward(r)
end

# ==== MATMUL ==== #

struct MatmulOp <: BinaryOp
    parent1::JNNode
    parent2::JNNode
    child::Tensor
end
function forward(op::MatmulOp)
    out_shape = (size(op.parent1.data)[1], size(op.parent2.data)[2])
    op.child.parent = op
    op.child.data = op.parent1.data * op.parent2.data
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::MatmulOp)
    grad1 = zeros(Matrix{Float32}(size(op.parent1.grad)...))
    grad2 = zeros(Matrix{Float32}(size(op.parent2.grad)...))
    grad1 += op.child.grad * transpose(op.parent2.data)
    grad2 += transpose(op.parent1.data) * op.child.grad
    return grad1, grad2
end
function *(t1::Tensor, t2::Tensor)
    op = MatmulOp(t1, t2, Tensor())
    return forward(op)
end
function *(m1::Matrix{Float32}, t2::Tensor)
    t1 = Tensor(m1, trainable=false)
    return t1 * t2
end
function *(t1::Tensor, m2::Matrix{Float32})
    t2 = Tensor(m2, trainable=false)
    return t1 * t2
end

# ==== ADD ==== #

struct AddOp <: BinaryOp
    parent1::JNNode
    parent2::JNNode
    child::Tensor
end
function forward(op::AddOp)
    op.child.parent = op
    op.child.data = op.parent1.data .+ op.parent2.data
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::AddOp)
    grad1 = zeros(Matrix{Float32}(size(op.parent1.grad)...))
    grad2 = zeros(Matrix{Float32}(size(op.parent2.grad)...))
    if size(grad1)[1] > size(grad2)[1]
        grad1 += op.child.grad
        grad2 .+= sum(op.child.grad, 1)
    elseif size(grad1)[1] < size(grad2)[1]
        grad1 .+= sum(op.child.grad, 1)
        grad2 += op.child.grad
    else
        grad1 += op.child.grad
        grad2 += op.child.grad
    end
    return grad1, grad2
end
function +(t1::Tensor, t2::Tensor)
    op = AddOp(t1, t2, Tensor())
    return forward(op)
end
function +(m1::Matrix{Float32}, t2::Tensor)
    t1 = Tensor(m1, trainable=false)
    return t1 + t2
end
function +(t1::Tensor, m2::Matrix{Float32})
    t2 = Tensor(m2, trainable=false)
    return t1 + t2
end

# ==== SUBTRACT ==== #

struct SubtractOp <: BinaryOp
    parent1::JNNode
    parent2::JNNode
    child::Tensor
end
function forward(op::SubtractOp)
    op.child.parent = op
    op.child.data = op.parent1.data .- op.parent2.data
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::SubtractOp)
    grad1 = zeros(Matrix{Float32}(size(op.parent1.grad)...))
    grad2 = zeros(Matrix{Float32}(size(op.parent2.grad)...))
    if size(grad1)[2] > size(grad2)[2]
        grad1 += op.child.grad
        grad2 .-= sum(op.child.grad, 2)
    elseif size(grad1)[2] < size(grad2)[2]
        grad1 .+= sum(op.child.grad, 2)
        grad2 -= op.child.grad
    else
        grad1 += op.child.grad
        grad2 -= op.child.grad
    end
    return grad1, grad2
end
function -(t1::Tensor, t2::Tensor)
    op = SubtractOp(t1, t2, Tensor())
    return forward(op)
end
function -(m1::Matrix{Float32}, t2::Tensor)
    t1 = Tensor(m1)
    return t1 - t2
end
function -(t1::Tensor, m2::Matrix{Float32})
    t2 = Tensor(m2, trainable=false)
    return t1 - t2
end

# ==== ABS ==== #

struct AbsOp <: UnaryOp
    parent::JNNode
    child::Tensor
end
function forward(op::AbsOp)
    op.child.parent = op
    op.child.data = abs.(op.parent.data)
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::AbsOp)
    same = convert(Array{Float32}, op.child.data .== op.parent.data)
    diff = 1 .- same
    return (op.child.grad .* same) - (op.child.grad .* diff)
end
function abs(t::Tensor)
    op = AbsOp(t, Tensor())
    return forward(op)
end

# ==== POWER ==== #

struct PowerOp <: UnaryOp
    parent::Tensor
    child::Tensor
    exponent::Float32
end
function forward(op::PowerOp)
    op.child.parent = op
    op.child.data = op.parent.data .^ op.exponent
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::PowerOp)
    return (op.exponent .* op.parent.data .^ (op.exponent - 1)) .* op.child.grad
end
function ^(t::Tensor, pow::Real)
    op = PowerOp(t, Tensor(), pow)
    return forward(op)
end

# ==== EXP ==== #

struct ExpOp <: UnaryOp
    parent::Tensor
    child::Tensor
end
function forward(op::ExpOp)
    op.child.parent = op
    op.child.data = exp.(op.parent.data)
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::ExpOp)
    return op.child.grad .* exp.(op.parent.data)
end
function exp(t::Tensor)
    op = ExpOp(t, Tensor())
    return forward(op)
end

# ==== RELU ==== #

struct ReluOp <: UnaryOp
    parent::Tensor
    child::Tensor
end
function forward(op::ReluOp)
    op.child.parent = op
    op.child.data = op.parent.data .* (op.parent.data .> 0)
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::ReluOp)
    return op.child.grad .* (op.parent.data .> 0)
end
function relu(t::Tensor)
    op = ReluOp(t, Tensor())
    return forward(op)
end

# ==== SOFTMAX ==== #

# function sotfmax(t::Tensor)
#     exps = exp()
# end
struct SoftmaxOp <: UnaryOp
    parent::Tensor
    child::Tensor
end
function forward(op::SoftmaxOp)
    op.child.parent = op
    shifted = op.parent.data .- maximum(op.parent.data, 2)
    exps = exp.(shifted)
    op.child.data = exps ./ sum(exps, 2)
    op.child.grad = zeros(Array{Float32}(size(op.child.data)))
    return op.child
end
function backward(op::SoftmaxOp)
    grad = zeros(Matrix{Float32}(size(op.parent.grad)...))
    dims = size(op.child.data)
    jacobian = Matrix{Float32}(dims[2], dims[2])
    for i in 1:dims[1]
        # Calculate the jacobian for row i
        jacobian[:, :] = .-(op.child.data[i, :] * transpose(op.child.data[i, :]))
        jacobian[:, :] .*= (1 .- eye(dims[2]))
        jacobian[:, :] .+= Diagonal(op.child.data[i, :] .* (1 .- op.child.data[i, :]))
        jacobian[:, :] .*= op.child.grad[i, :]

        # # Apply the jacobian
        grad[i:i, :] .+= sum(jacobian[:, :], 1)
    end
    return grad
end
function softmax(t::Tensor)
    op = SoftmaxOp(t, Tensor())
    return forward(op)
end

# -----------------------------OPTIM------------------------------------------

function apply_and_zero_gradients(t::Tensor)
    apply_gradients!(t)
    zero_gradients!(t)
    apply_and_zero_gradients(t.parent)
end
function apply_and_zero_gradients(op::BinaryOp)
    apply_and_zero_gradients(op.parent1)
    apply_and_zero_gradients(op.parent2)
end
function apply_and_zero_gradients(op::UnaryOp)
    apply_and_zero_gradients(op.parent)
end
function apply_and_zero_gradients(n::None)
end

abstract type Optimizer end

# ==== SGD ==== #

struct SGDOptimizer <: Optimizer
end
function optimize!(optim::SGDOptimizer, t::Tensor; step_size::Real=0.01)
    function optim_inner(t::Tensor)
        optim_inner(t.parent)
    end
    function optim_inner(op::BinaryOp)
        grad1, grad2 = backward(op)
        op.parent1.grad += grad1
        op.parent2.grad += grad2
        optim_inner(op.parent1)
        optim_inner(op.parent2)
    end
    function optim_inner(op::UnaryOp)
        grad = backward(op)
        op.parent.grad += grad
        optim_inner(op.parent)
    end
    function optim_inner(n::None)
    end
    fill!(t.grad, step_size)
    optim_inner(t.parent)
    apply_and_zero_gradients(t.parent)
end

# ==== SGD With Momentum ==== #

mutable struct SGDWithMomentum <: Optimizer
    momentums::Dict{Tensor, Matrix{Float32}}
    gamma::AbstractFloat
end
SGDWithMomentum() = SGDWithMomentum(Dict{Tensor, Matrix{Float32}}(), 0.5)
function SGDWithMomentum(momentum::AbstractFloat)
    return SGDWithMomentum(Dict{Tensor, Matrix{Float32}}(), momentum)
end
function optimize!(optim::SGDWithMomentum, t::Tensor; step_size::Real=0.01)
    function optim_inner(t::Tensor)
        optim_inner(t.parent)
    end
    function optim_inner(op::BinaryOp)
        if !(op.parent1 in keys(optim.momentums))
            println("Creating momentum matrix")
            optim.momentums[op.parent1] = Matrix{Float32}(size(op.parent1.grad)...)
        end
        if !(op.parent2 in keys(optim.momentums))
            optim.momentums[op.parent2] = Matrix{Float32}(size(op.parent2.grad)...)
        end

        grad1, grad2 = backward(op)
        optim.momentums[op.parent1] = optim.gamma .* optim.momentums[op.parent1] + grad1
        optim.momentums[op.parent2] = optim.gamma .* optim.momentums[op.parent2] + grad2
        op.parent1.grad += optim.momentums[op.parent1]
        op.parent2.grad += optim.momentums[op.parent2]
        optim_inner(op.parent1)
        optim_inner(op.parent2)
    end
    function optim_inner(op::UnaryOp)
        if !(op.parent in keys(optim.momentums))
            optim.momentums[op.parent] = Matrix{Float32}(size(op.parent.grad)...)
        end

        grad = backward(op)
        optim.momentums[op.parent] = optim.gamma .* optim.momentums[op.parent] + grad
        op.parent.grad += optim.momentums[op.parent]
        optim_inner(op.parent)
    end
    function optim_inner(n::None)
    end
    fill!(t.grad, step_size)
    optim_inner(t.parent)
    apply_and_zero_gradients(t.parent)
end

# -----------------------------HELPERS--------------------------------------- #

function fc_layer(num_inputs::Integer, num_outputs::Integer;
                  init_fn::Function=variance_scaled_tensor,
                  activation_fn::Function=relu)
    W = init_fn(num_inputs, num_outputs)
    b = init_fn(1, num_outputs)
    f(t::Union{Tensor, Matrix{Float32}}) = activation_fn((t * W) + b)
    return f, (W, b)
end

end
