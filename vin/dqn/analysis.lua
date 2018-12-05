if not dqn then
    require 'initenv'
end

require 'vin_gvgai'
require 'text'


--- file to perform some analysis
manifold = require 'manifold'
net = torch.load(arg[1])

-- get the embedding module. 
embedding_module = net.model:findModules('nn.LookupTableMaskZero')[1]

normalize=false

-- Get vectors and normalize.
vec_size = nil
vec = {}
for k, v in pairs(net.word_to_int) do 
    num = tonumber(k)
    if not num then                    
        vec[k] = embedding_module:forward(torch.ones(1)*v):squeeze()                
        local norm = vec[k]:norm()
        if normalize and norm > 0 then
            vec[k]:div(norm)
        end    
        vec_size = vec_size or vec[k]:size(1)
    end
    
end

-- print(vec)

function dot(a, b)
    return torch.dot(vec[a], vec[b])
end

function nearest_neighbors()
    for i, v in pairs(vec) do
        local maxDot = -10
        local NN = i
        for j, w in pairs(vec) do
            if j ~= i then
                if torch.dot(v,w) > maxDot then
                    maxDot = torch.dot(v,w)
                    NN = j
                end
            end
        end
        print(i, NN ,maxDot)
    end
end

function find_len(table)
    local cnt = 0
    for k, v in pairs(table) do
        cnt = cnt+1
    end
    return cnt
end

function plot_tsne(vec)
    local n = find_len(vec)
    local m = torch.zeros(n, vec_size)
    local i = 1
    local symbols = {}        
    for k, val in pairs(vec) do
        if k ~= 'NULL' then       
            m[i] = val
            symbols[i] = k
            i = i+1            
        end
    end  
  
  opts = {ndims = 2, perplexity = 50, pca = 50, use_bh = false}
  --opts = {ndims = 2, perplexity =20}
  mapped_x1 = manifold.embedding.tsne(m)
  return mapped_x1, symbols
end

tsne, symbols = plot_tsne(vec)
--write
local file = io.open('tsne.txt', "w");
for i=1, #symbols do
    file:write(symbols[i] .. ' ' .. tsne[i][1]  .. ' ' .. tsne[i][2] .. '\n')
end




