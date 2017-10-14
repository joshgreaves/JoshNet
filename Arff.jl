module Arff

export ArffFile
export loadarff

mutable struct ArffFile
    attributes::Array{String}
    data::Array{Any}

    function ArffFile()
        return new(Array{String}(0), Array{Any}(0))
    end
end

function addattribute!(f::ArffFile, attr::String)
    push!(f.attributes, attr)
end

function adddata!(f::ArffFile, data::Array{<:Any, 2})
    f.data = [f.data; data]
end

function loadarff(path::String; mappings::Dict{<:Any, <:Any}=Dict(),
                  should_parse::Bool=true)
    lines = Array{String}(0)
    open(path) do f
        for line in readlines(f)
            if length(line) > 0 && line[1] != '%'
                push!(lines, line)
            end
        end
    end

    result = ArffFile()
    for i in 1:length(lines)
        if startswith(lines[i], "@ATTRIBUTE") || startswith(lines[i], "@attribute")
            addattribute!(result, lines[i][(length("@ATTRIBUTE") + 2):end])
        elseif startswith(lines[i], "@DATA") || startswith(lines[i], "@data")
            for j in (i+1):length(lines)
                if !startswith(lines[j], '%')
                    split_line = split(lines[j], ',')
                    vec = map(x -> get(mappings, x, x), split_line)
                    if should_parse
                        vec = parse.(vec)
                    end
                    adddata!(result, reshape(vec, 1, length(vec)))
                end
            end
            break
        end
    end
    return result
end
end
