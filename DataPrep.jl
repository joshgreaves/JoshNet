module DataPrep

function splitdata(data::Array{<:Any, 2}, split::AbstractFloat=0.75)
    data_points = size(data)[1]
    indices = shuffle(1:data_points)
    split_index = Int32(floor(data_points * split))
    return (data[indices[1:split_index],:], data[indices[(split_index+1):end],:])
end

function splitdata(data::Array{<:Any, 2}, labels::Array{<:Any, 2},
                   split::AbstractFloat=0.75)
    data_points = size(data)[1]
    indices = shuffle(1:data_points)
    split_index = Int32(floor(data_points * split))
    return data[indices[1:split_index],:], labels[indices[1:split_index],:],
        data[indices[(split_index+1):end],:], labels[indices[(split_index+1):end],:]
end

function shuffledata(data::Array{<:Any, 2})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:]
end

function shuffledata(data::Array{<:Any, 2}, labels::Array{<:Any, 1})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:], labels[indices]
end

function shuffledata(data::Array{<:Any, 2}, labels::Array{<:Any, 2})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:], labels[indices,:]
end

function getbatch(data::Matrix{<:Any}, labels::Matrix{<:Any};
                  batch_size::Integer=32)
    num_data = size(data)[1]
    indices = shuffle(1:num_data)
    return data[indices[1:batch_size], :], labels[indices[1:batch_size], :]
end

function smoothline(data::Array{<:Number, 1}; window=10)
    result = Array{Float64, 1}(Int32(floor(size(data)[1] / 10)))
    for i in 1:length(result)
        result[i] = mean(data[((i-1)*window+1):(i*window)])
    end
    return result
end

end
