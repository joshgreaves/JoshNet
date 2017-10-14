import Base.*
import Base.+
import Base.-
import Base.abs

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
Tensor() = Tensor(Matrix{Float32}(0, 0), Matrix{Float32}(0, 0), None(), true)
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
function backward(t::Tensor)
    backward(t.parent)
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

function gaussian_tensor(height::Integer, width::Integer; trainable::Bool=true)
    return Tensor(convert(Array{Float32, 2}, randn(height, width)), trainable=trainable)
end

# ------------------------- OPS ----------------------------------------------

# ==== REDUCE MEAN ==== #
struct ReduceMeanOp <: UnaryOp
    parent::JNNode
    child::JNNode
end
function forward(op::ReduceMeanOp)
    op.child.parent = op
    op.child.data = Array{Float32}(1, 1)
    op.child.data[1, 1] = mean(op.parent.data)
    op.child.grad = zeros(Array{Float32}(1, 1))
    return op.child
end
function backward(op::ReduceMeanOp)
    op.parent.grad += (1 / length(op.parent.grad)) .* op.child.grad[1, 1]
end

function reduce_mean(t1::Tensor)
    r = ReduceMeanOp(t1, Tensor())
    return forward(r)
end

# ==== MATMUL ==== #

mutable struct MatmulOp <: BinaryOp
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
    op.parent1.grad += op.child.grad * transpose(op.parent2.data)
    op.parent2.grad += transpose(op.parent1.data) * op.child.grad
end
function *(t1::Tensor, t2::Tensor)
    op = MatmulOp(t1, t2, Tensor())
    return forward(op)
end

# ==== ADD ==== #

mutable struct AddOp <: BinaryOp
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
    if size(op.parent1.grad)[1] > size(op.parent2.grad)[1]
        op.parent1.grad += op.child.grad
        op.parent2.grad .+= sum(op.child.grad, 1)
    elseif size(op.parent1.grad)[1] < size(op.parent2.grad)[1]
        op.parent1.grad .+= sum(op.child.grad, 1)
        op.parent2.grad += op.child.grad
    else
        op.parent1.grad += op.child.grad
        op.parent2.grad += op.child.grad
    end
end
function +(t1::Tensor, t2::Tensor)
    op = AddOp(t1, t2, Tensor())
    return forward(op)
end

# ==== SUBTRACT ==== #

mutable struct SubtractOp <: BinaryOp
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
    if size(op.parent1.grad)[2] > size(op.parent2.grad)[2]
        op.parent1.grad += op.child.grad
        op.parent2.grad .-= sum(op.child.grad, 2)
    elseif size(op.parent1.grad)[2] < size(op.parent2.grad)[2]
        op.parent1.grad .+= sum(op.child.grad, 2)
        op.parent2.grad -= op.child.grad
    else
        op.parent1.grad += op.child.grad
        op.parent2.grad -= op.child.grad
    end
end
function -(t1::Tensor, t2::Tensor)
    op = SubtractOp(t1, t2, Tensor())
    return forward(op)
end

# ==== ABS ==== #

mutable struct AbsOp <: UnaryOp
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
    op.parent.grad += (op.child.grad .* same) - (op.child.grad .* diff)
end
function abs(t::Tensor)
    op = AbsOp(t, Tensor())
    return forward(op)
end

# ==== RELU ==== #

mutable struct ReluOp <: UnaryOp
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
    op.parent.grad = op.child.grad .* (op.parent.data .> 0)
end
function relu(t::Tensor)
    op = ReluOp(t, Tensor())
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

function sgd_optimizer(t::Tensor; step_size::Real=0.01)
    function sgd_optim_inner(t::Tensor)
        sgd_optim_inner(t.parent)
    end
    function sgd_optim_inner(op::BinaryOp)
        backward(op)
        sgd_optim_inner(op.parent1)
        sgd_optim_inner(op.parent2)
    end
    function sgd_optim_inner(op::UnaryOp)
        backward(op)
        sgd_optim_inner(op.parent)
    end
    function sgd_optim_inner(n::None)
    end
    fill!(t.grad, step_size)
    sgd_optim_inner(t.parent)
    apply_and_zero_gradients(t.parent)
end
