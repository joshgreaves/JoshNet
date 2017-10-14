include("./joshnet.jl")

input = gaussian_tensor(10, 2, trainable=false) # 10 instances of 2 points
labels = Tensor(convert(Matrix{Float32}, sum(input.data, 2)), trainable=false)
learning_rate = 0.1

W1 = gaussian_tensor(2, 10)
b1 = gaussian_tensor(1, 10)

W2 = gaussian_tensor(10, 10)
b2 = gaussian_tensor(1, 10)

W3 = gaussian_tensor(10, 1)
b3 = gaussian_tensor(1, 1)

println(W1.data)
for i in 1:100000
    h1 = relu((input * W1) + b1)
    h2 = relu((h1 * W2) + b2)
    o = (h2 * W3) + b3
    loss = reduce_mean(abs(o - labels))
    if i % 1 == 0
        println(i, ": ", loss.data[1, 1])
    end

    sgd_optimizer(loss)
end
println(W1.data)
