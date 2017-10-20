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
learning_rate = 0.001
num_epochs = 1000
batch_size = 1

runs = 10
train_mse = Matrix{Float32}(runs, 1)
val_mse = Matrix{Float32}(runs, 1)
val_acc = Matrix{Float32}(runs, 1)
stopping_time = Matrix{Int32}(runs, 1)

for k in 1:runs
    println(k)

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

    # Store data, and weights for stopping criteria
    steps_since_update = 0
    most_steps = 20
    best_mse = 1000
    cached_wb1 = deepcopy(Wb1)
    cached_wb2 = deepcopy(Wb2)
    num_steps = 0
    stopping_index = 0
    function cache_model()
        cached_wb1[1].data = copy(Wb1[1].data)
        cached_wb1[2].data = copy(Wb1[2].data)
        cached_wb2[1].data = copy(Wb2[1].data)
        cached_wb2[2].data = copy(Wb2[2].data)
    end

    for i in 1:num_epochs
        # Do a training step
        shuffled_x, shuffled_y = DataPrep.shuffledata(train_x, train_y)
        for j in 1:train_size
            o = classify(shuffled_x[j:j, :])
            loss = reduce_mean(reduce_sum((o - shuffled_y[j:j, :])^2.0, axis=[2]))

            optimize!(optim, loss, step_size=learning_rate)
        end

        this_mse = evaluate_mse(val_x, val_y)
        num_steps += 1
        steps_since_update += 1

        if this_mse < best_mse
            best_mse = this_mse
            cache_model()
            steps_since_update = 0
            stopping_index = i
        elseif steps_since_update == most_steps
            break
        end
    end

    # Set the weights back to where they should be
    Wb1[1].data = cached_wb1[1].data
    Wb1[2].data = cached_wb1[2].data
    Wb2[1].data = cached_wb2[1].data
    Wb2[2].data = cached_wb2[2].data


    train_mse[k, 1] = evaluate_mse(train_x, train_y)
    val_mse[k, 1] = evaluate_mse(val_x, val_y)
    val_acc[k, 1], _ = evaluate(val_x, val_y)
    stopping_time[k, 1] = stopping_index
end

println("Average train: ", mean(train_mse))
println("Average val: ", mean(val_mse))
println("Average acc: ", mean(val_acc))
println("Stopping time: ", mean(stopping_time))
