
require "util.initenv"

-- override (zero out outputs for input id 1 (this is actually for
-- input 0, which we increment by 1 before passing to lookuptable
-- in the network below))
function nn.LookupTable:updateOutput(input)
   self:backCompatibility()
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end

     --zero out index 1
    local output = self.output:clone()
    for i=1, input:size(1) do
        if input[i] == 1 then --zero out index 1
            output[i]:mul(0)
        end
    end

    self.output = output

   return self.output
end

function create_network(args)

    local net = nn.Sequential()
    -- Add 1 to avoid passing 0 as input to lookuptable.
    net:add(nn.AddConstant(1, true))

    net:add(nn.Reshape(args.objects_per_cell, args.input_dims[1] * args.input_dims[2] *
                       args.input_dims[3]))

    -- Split table and then pass through lookuptable.
    net:add(nn.SplitTable(2))
    
    -- First, use a lookuptable to transform the input
    local map = nn.MapTable()
    local emb_dim = 3
    map:add(nn.LookupTable(1000, emb_dim))  -- 3-dim embeddings
    -- map:add(nn.Transpose({2, 3}))
    net:add(map)


    -- Add tables.
    net:add(nn.CAddTable())

    net:add(nn.Transpose({2, 3}))

    local conv_input_dims = {args.input_dims[1] * emb_dim, 
                       args.input_dims[2],
                       args.input_dims[3]}
    net:add(nn.Reshape(unpack(conv_input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolutionMM

    net:add(convLayer(emb_dim * args.hist_len * args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end

    -- Find the output dimension.
    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.ones(1, args.objects_per_cell, unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.ones(1, args.objects_per_cell, unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(last_layer_size, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
	  if args.cudnn and args.cudnn > 0 then
       -- cuDNN-ify the network
       cudnn.convert(net, cudnn)
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    
    return net
end
