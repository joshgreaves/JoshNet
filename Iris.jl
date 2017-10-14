push!(LOAD_PATH, ".")

importall JoshNet
import DataPrep
import Arff

# Load the Iris dataset and get it into the correct form
arff = Arff.loadarff("data/iris.arff")
data = convert(Matrix{Float32}, arff.data[:, 1:end-1])

num_data = size(data)[1]
num_features = size(data)[2]
num_classes = 3

mappings = Dict(:(Iris - setosa) =>  Float32[1.0, 0.0, 0.0],
                :(Iris - versicolor) => Float32[0.0, 1.0, 0.0],
                :(Iris - virginica) => Float32[0.0, 0.0, 1.0])
labels = Matrix{Float32}(num_data, num_classes)
for i in 1:num_data
    labels[i, :] = mappings[arff.data[i, end]]
end

# Define the weights
W1 = gaussian_tensor(4, 20)
b1 = gaussian_tensor(1, 20)

W2 = gaussian_tensor(20, 20)
b2 = gaussian_tensor(1, 20)

W3 = gaussian_tensor(20, num_classes)
b3 = gaussian_tensor(1, num_classes)

# Define hyperparameters
learning_rate = 0.00001
num_epochs = 100
batch_size = 10

# Train
for i in 1:1000
    # Get a batch
    batch_x, batch_y = DataPrep.getbatch(data, labels, batch_size=batch_size)

    h1 = relu((batch_x * W1) + b1)
    # h2 = relu((h1 * W2) + b2)
    o = softmax((h1 * W3) + b3)
    loss = reduce_mean(abs(o - batch_y))
    if i % 1 == 0
        println(i, ": ", loss.data[1, 1])
        # println(batch_y, o.data)
    end

    sgd_optimizer(loss, step_size=learning_rate)
end
