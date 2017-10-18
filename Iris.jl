include("./JoshNet.jl")
include("./DataPrep.jl")
include("./Arff.jl")

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

# ============================== NETWORK DEF ================================ #

# Hyperparameters
learning_rate = 0.1
num_epochs = 1000
batch_size = 32

# Layers
fc1, Wb1 = fc_layer("Layer1", 4, 100)
fc2, Wb2 = fc_layer("Layer2", 100, 100)
fc3, Wb3 = fc_layer("Layer3", 100, num_classes, activation_fn=softmax)
optim = SGDWithMomentum()

# The network definition
function classify(input::Matrix{Float32})
    h1 = fc1(input)
    h2 = fc2(h1)
    return fc3(h2)
end

# =============================== TRAINING ================================== #
# Calculate overall accuracy
function evaluate()
    predictions = classify(data)
    p_maxvals, p_maxindices = findmax(predictions.data, 2)
    t_maxvals, t_maxindices = findmax(labels, 2)
    correct = sum(p_maxindices .== t_maxindices)
    incorrect = num_data - correct
    println("Accuracy: ", correct, "/", num_data, " = ", correct/num_data)
    println("Error: ", incorrect, "/", num_data, " = ", incorrect/num_data)
end
evaluate()

for i in 1:num_epochs
    # Get a batch
    batch_x, batch_y = DataPrep.getbatch(data, labels, batch_size=batch_size)
    o = classify(batch_x)
    loss = reduce_mean(reduce_sum((o - batch_y)^2.0, axis=[2]))

    if i % 20 == 0
        println(i, ": ", loss.data[1, 1])
    end

    optimize!(optim, loss, step_size=learning_rate)
end


evaluate()
