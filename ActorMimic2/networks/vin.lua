require 'nngraph'
-- require 'ValueFilter'
-- require 'ObjectFilter'
-- require 'ObjectDescriptor'
-- require 'rnn'

local nninit = require 'nninit'

local bias_init = 0
local reactive_bias_init = 0
local batch_norm = false  -- for convnets
local batch_linear_norm = false  -- for linear layers
local mean_pooling = false

local emb_dim = 20  -- Embedding

nngraph.setDebug(true)


function calculate_output_size(height, width, kW, kH, dW, dH, padW, padH)
    -- Calculate the output size of a convolution. 
    owidth  = math.floor((width + 2*padW - kW) / dW) + 1
    oheight = math.floor((height + 2*padH - kH) / dH) + 1
    return oheight, owidth
end

-- The VIN.
function create_network(args)
    -- local S_input = nn.AddConstant(1)()  -- input state of dimension (batch_size x )
    -- local text = nn.AddConstant(1)()

    args.vin_k = 0  -- Only using reactive model for AMN net.

    local S_input = nn.Identity()()  -- input state of dimension (batch_size x )
    -- local text = nn.Identity()()


    local emb_layer = nn.LookupTableMaskZero(1000, emb_dim)

    local description_length
    local text_embedding

    -- if args.object_filter then
    --     local description_text = nn.Narrow(2, 2, -1)(text)  -- ignore first column.
    --     description_length = max_sent_length - 1
    --     text_embedding = emb_layer(description_text)
    -- else
    --     description_length = max_sent_length
    --     text_embedding = emb_layer(text)
    -- end

    -- local lstm_out_dim = emb_dim
    -- local seqlstm = nn.SeqLSTM(emb_dim, lstm_out_dim)
    -- --local seqlstm = nn.SeqBRNN(emb_dim, lstm_out_dim)  -- bidirectional
    -- seqlstm.batchfirst = true
    -- seqlstm.maskzero = true

    -- -- This is the representation of all the text sentences.
    -- local X_lstm = seqlstm(text_embedding) 

    -- -- Convert this into convnet input representation. 
    -- -- X = nn.Transpose({2,3}, {1,2})(X)

    -- -- Get the ids.
    -- if args.object_filter then
    --     object_ids = nn.Select(2, 1)(text)  -- object ids are the first word in every sentence.
    -- end

    -- local X
    -- -- Mean pooling. TODO: check embedding and reshape compatibility.
    -- if mean_pooling then
    --     X = nn.Sum(2)(X_lstm)
    --     X = nn.Contiguous()(X)
    --     X = nn.View(-1, lstm_out_dim * max_sentences)(X)
    --     X = nn.Linear(lstm_out_dim * max_sentences,
    --                   lstm_out_dim * args.input_dims[2] * args.input_dims[3])(X)
    -- else
    --     X = nn.Contiguous()(X_lstm)
    --     X = nn.View(-1, lstm_out_dim * max_sentences * description_length)(X)
    --     X = nn.Linear(lstm_out_dim * max_sentences * description_length,
    --                   lstm_out_dim * args.input_dims[2] * args.input_dims[3])(X)
    -- end
    
    -- -- Convert to match state dimensions. 
    -- X = nn.Reshape(lstm_out_dim, args.input_dims[2], args.input_dims[3])(X)

    -- To remove text dependence. 
    -- X = nn.MulConstant(0)(X)
    
    -- process the state and convert it to embedding form.
    local S_input_reshaped = nn.Reshape(args.objects_per_cell * args.input_dims[1] * args.input_dims[2] * args.input_dims[3])(S_input)
    
    local S = emb_layer:clone('weight', 'bias', 'gradWeight', 'gradBias')(S_input_reshaped)

    -- replace object embeddings using their text descriptions. 
    -- if args.object_filter then
    --     X_mean = nn.Sum(2)(X_lstm)
    --     S_descriptor = nn.ObjectDescriptor(emb_dim, max_sentences)({S_input_reshaped, S, object_ids, X_mean})        
    --     -- merge both representations. 
    --     S = nn.JoinTable(3)({S, S_descriptor})  -- last dim is embedding dim
    --     --S = nn.JoinTable(3)({S, nn.MulConstant(0)(S_descriptor)})  -- last dim is embedding dim
       
    --     -- Use this for completely replacing object vector with that derived from text descriptions.
    --     --S = nn.ObjectFilter(emb_dim, max_sentences)({S_input_reshaped, S, object_ids, X_mean})

    --     -- since emb_dim is now doubled
    --     emb_dim = 2 * emb_dim

    -- end

    -- Now, we have a matrix of size {state_dim, emb_dim}

    -- First reshape such that related embeddings are grouped properly.
    S = nn.Reshape(args.input_dims[1], args.objects_per_cell,
                   args.input_dims[2],
                   args.input_dims[3], emb_dim)(S)

    -- Add up across the objects dimension.
    S = nn.Sum(3)(S)

    -- Now, the dimensions are (bsize x input[1] x input[2] x input[3] x emb_dim)
    -- Now, transpose (remember that dim 1 is batch dim).
    S = nn.Transpose({3,5}, {4,5})(S)

    -- Now, the dimensions are (bsize x input[1] x emb_dim x input[2] x input[3])
    S = nn.Reshape(args.input_dims[1] * emb_dim, args.input_dims[2],
                   args.input_dims[3])(S)

    -- local R = nn.SpatialConvolution(args.input_dims[1] * emb_dim, 
    --                                 args.r_channels[1], 
    --                                 args.r_filter[1], args.r_filter[2],
    --                                 args.r_filter_stride[1], args.r_filter_stride[2],
    --                                 1, 1)
    --                                 :init('weight', nninit.kaiming, {gain = 'relu'})
    --                                 :init('bias', nninit.constant, bias_init)(S)
    -- if batch_norm then
    --     R = nn.SpatialBatchNormalization(args.r_channels[1])
    --                                     :init('weight', nninit.normal, 1.0, 0.002)
    --                                     :init('bias', nninit.constant, 0)(R)
    -- end

    -- for i=2, #args.r_channels do
    --     R = nn.SpatialConvolution(args.r_channels[i-1], 
    --                                 args.r_channels[i], 
    --                                 args.r_filter[1], args.r_filter[2],
    --                                 args.r_filter_stride[1], args.r_filter_stride[2],
    --                                 1, 1)
    --                                 :init('weight', nninit.kaiming, {gain = 'relu'})
    --                                 :init('bias', nninit.constant, bias_init)(R)
    --     if batch_norm then
    --         R = nn.SpatialBatchNormalization(args.r_channels[i])
    --                                     :init('weight', nninit.normal, 1.0, 0.002)
    --                                     :init('bias', nninit.constant, 0)(R)
    --     end
    -- end

    -- -- pad the reward.
    -- local padded_R = nn.Padding(2, 1)(R)

    -- -- Convolution for Q
    -- local Qconv = nn.SpatialConvolution(args.r_channels[#args.r_channels] + 1, 
    --                                 args.n_actions, 
    --                                 args.q_filter[1], args.q_filter[2],
    --                                 args.q_filter_stride[1], args.q_filter_stride[2],
    --                                 1, 1)
    --                                 :init('weight', nninit.kaiming, {gain = 'relu'})
    --                                 :init('bias', nninit.constant, bias_init)

    -- local Q = Qconv(padded_R)

    -- if batch_norm then
    --     Q = nn.SpatialBatchNormalization(args.n_actions)
    --                                     :init('weight', nninit.normal, 1.0, 0.002)
    --                                     :init('bias', nninit.constant, 0)(Q)
    -- end

    -- local V = nn.Max(2)(Q)

    -- -- Now, do recurrent value iteration.
    -- for i=1, args.vin_k - 1 do
    --     V = nn.Reshape(1, args.input_dims[2], args.input_dims[3], true)(V)
    --     C = nn.JoinTable(2)({R, V})
    --     Qconv_copy = Qconv:clone('weight', 'bias', 'gradWeight', 'gradBias')  -- share the params
    --     Q = Qconv_copy(C)

    --     if batch_norm then
    --         Q = nn.SpatialBatchNormalization(args.n_actions)
    --                                     :init('weight', nninit.normal, 1.0, 0.002)
    --                                     :init('bias', nninit.constant, 0)(Q)
    --     end

    --     V = nn.Max(2)(Q)

    --     --non-linearity
    --     V = nn.Sigmoid()(V)
    -- end

    -- -- Last convolution
    -- V = nn.Reshape(1, args.input_dims[2], args.input_dims[3], true)(V)
    -- C = nn.JoinTable(2)({R, V})
    -- Qconv_copy = Qconv:clone('weight', 'bias', 'gradWeight', 'gradBias')  -- share the params
    -- Q_unfiltered = Qconv_copy(C)

    -- -- Filter to get the Q value for the current position. 
    -- local S_reshaped = nn.Reshape(args.input_dims[1] * args.objects_per_cell, 
    --                args.input_dims[2],
    --                args.input_dims[3])(S_input)
    -- Q = nn.ValueFilter(1)({S_reshaped, Q_unfiltered})  -- agent_id = 2, since we have added 1 to all cells in state.

    -- Reactive module.

    -- set the initial height and width
    local height = args.input_dims[2]
    local width = args.input_dims[3]

    local Q_reactive = nn.SpatialConvolution(
            args.input_dims[1] * emb_dim,             
            args.n_units[1], 
            args.filter_size[1], args.filter_size[1],
            args.filter_stride[1], args.filter_stride[1],
            1,1)
            :init('weight', nninit.kaiming, {gain = 'relu'})
            :init('bias', nninit.constant, reactive_bias_init)(S)

    if batch_norm then
        Q_reactive = nn.SpatialBatchNormalization(args.n_units[1])
                                        :init('weight', nninit.normal, 1.0, 0.002)
                                        :init('bias', nninit.constant, 0)(Q_reactive)
    end


    height, width = calculate_output_size(height, width, args.filter_size[1], args.filter_size[1], args.filter_stride[1], args.filter_stride[1], 1, 1)
    Q_reactive = args.nl()(Q_reactive)

    -- add convolutional layers. 
    for i=1, (#args.n_units-1) do
        Q_reactive = nn.SpatialConvolution(
            args.n_units[i],             
            args.n_units[i+1],
            args.filter_size[i+1], args.filter_size[i+1],
            args.filter_stride[i+1], args.filter_stride[i+1],
            1,1)
            :init('weight', nninit.kaiming, {gain = 'relu'})
            :init('bias', nninit.constant, reactive_bias_init)(Q_reactive)
        if batch_norm then
            Q_reactive = nn.SpatialBatchNormalization(args.n_units[i+1])
                                        :init('weight', nninit.normal, 1.0, 0.002)
                                        :init('bias', nninit.constant, 0)(Q_reactive)
        end

        height, width = calculate_output_size(height, width, args.filter_size[i+1], args.filter_size[i+1], args.filter_stride[i+1], args.filter_stride[i+1], 1, 1)
        Q_reactive = args.nl()(Q_reactive)
    end

     -- Find the output dimension. have to compute it explicitly.
    local nel

    nel = args.n_units[#args.n_units] * height * width
    Q_reactive = nn.Reshape(nel)(Q_reactive)


    -- add some linear layers.
    Q_reactive = nn.Linear(nel, args.n_hid[1])
                          :init('weight', nninit.xavier, {dist = 'normal'})(Q_reactive)
    if batch_linear_norm then
        Q_reactive = nn.BatchNormalization(args.n_hid[1])(Q_reactive)
    end


    Q_reactive = args.nl()(Q_reactive)
    last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        Q_reactive = nn.Linear(args.n_hid[i], last_layer_size)
                              :init('weight', nninit.xavier, {dist = 'normal'})(Q_reactive)
        if batch_linear_norm then
            Q_reactive = nn.BatchNormalization(last_layer_size)(Q_reactive)
        end

        Q_reactive = args.nl()(Q_reactive)
    end
    Q_reactive = nn.Linear(last_layer_size, args.n_actions)
                          :init('weight', nninit.xavier, {dist = 'normal'})(Q_reactive)

    -- Now, combine the Q predictions of both networks. 
    -- Q_final = nn.CAddTable()({Q, Q_reactive})

    
    print("-------------- reactive --------------------")
    net = nn.gModule({S_input}, {Q_reactive})  -- inputs and outputs of net        

    nngraph.annotateNodes()
    net.name = 'vin_text'
    return net  -- remember that the net output is over all positions.

end
