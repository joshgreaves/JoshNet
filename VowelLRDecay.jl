include("./JoshNet.jl")
include("./DataPrep.jl")
include("./Arff.jl")

using Plots

importall JoshNet
import DataPrep
import Arff

# Load the vowel dataset and get it into the correct form
arff = Arff.loadarff("data/vowel.arff")

# To help creating the dictionaries
function onehot(i::Integer, n::Integer)
    result = zeros(Float32, n)
    result[i] = 1.0
    return result
end

# Dicts for converting data into correct form
names = [:Andrew, :Bill, :David, :Mark, :Jo, :Kate, :Penny, :Rose, :Mike,
         :Nick, :Rich, :Tim, :Sarah, :Sue, :Wendy]
num_names = length(names)
name_to_id = Dict(names[i] => onehot(i, num_names) for i in 1:num_names)
gender_dict = Dict(:Male => Float32[1, 0], :Female => [0, 1])
class_names = [:hid, :hId, :hEd, :hAd, :hYd, :had, :hOd, :hod, :hUd, :hud, :hed]
num_classes = length(class_names)
label_dict = Dict(class_names[i] => onehot(i, num_classes) for i in 1:num_classes)

num_data = size(arff.data)[1]
num_features = num_names + 2 + 10 # 2 = male or female, 10 = feature0 -> feature10
data = Matrix{Float32}(num_data, num_features)
labels = Matrix{Float32}(num_data, num_classes)
for i in 1:num_data
    data[i, 1:num_names] = name_to_id[arff.data[i, 2]]
    data[i, num_names+1:num_names+2] = gender_dict[arff.data[i, 3]]
    data[i, num_names+3:num_names+12] = arff.data[i,4:end-1]
    labels[i, :] = label_dict[arff.data[i, end]]
end

# Split the dataset into train, validate, and test
train_x, train_y, test_x, test_y = DataPrep.splitdata(data, labels)
train_x, train_y, val_x, val_y = DataPrep.splitdata(train_x, train_y)
train_size = size(train_x)[1]
val_size = size(val_x)[1]
test_size = size(test_x)[1]

# ============================== NETWORK DEF ================================ #

# Hyperparameters
initial_learning_rate = 0.1
final_learning_rate = 0.001
num_epochs = 100

# Layers
fc1, Wb1 = fc_layer("Layer1", num_features, 2 * num_features)
fc2, Wb2 = fc_layer("Layer2", 2 * num_features, num_classes, activation_fn=softmax)
optim = SGDOptimizer()

# The network definition
function classify(input::Matrix{Float32})
    h1 = fc1(input)
    return fc2(h1)
end

# =============================== TRAINING ================================== #
# Calculate the evaluation set accuracy and loss
function evaluate(x, y)
    n = size(x)[1]
    predictions = classify(x)
    p_maxvals, p_maxindices = findmax(predictions.data, 2)
    t_maxvals, t_maxindices = findmax(y, 2)
    correct = sum(p_maxindices .== t_maxindices)
    incorrect = n - correct
    return correct / n, incorrect / n
end

function evaluate_mse(x, y)
    n = size(x)[1]
    predictions = classify(x)
    return reduce_mean((predictions - y)^2.0).data[1, 1]
end

train_mse = Vector{Float32}(num_epochs)
val_mse = Vector{Float32}(num_epochs)

for i in 1:num_epochs
    lr1 = ((num_epochs - i) / num_epochs) * initial_learning_rate
    lr2 = (i / num_epochs) * final_learning_rate
    learning_rate =  lr1 + lr2

    # Do a training step
    shuffled_x, shuffled_y = DataPrep.shuffledata(train_x, train_y)
    for j in 1:train_size
        o = classify(shuffled_x[j:j, :])
        loss = reduce_mean(reduce_sum((o - shuffled_y[j:j, :])^2.0, axis=[2]))

        optimize!(optim, loss, step_size=learning_rate)
    end

    train_mse[i] = evaluate_mse(train_x, train_y)
    val_mse[i] = evaluate_mse(val_x, val_y)
end

function plot_eval()
    pyplot()
    plot(train_mse, color=:blue, line=:solid, label="training mse")
    plot!(val_mse, color=:blue, line=:dash, label="validation mse")
    title!("Training on the Vowels Dataset with LR Decay")
    xaxis!("Epochs")
    yaxis!("MSE")
    # savefig("img/VowelLRDecay.png")
end
