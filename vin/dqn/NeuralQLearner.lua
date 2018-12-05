--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'initenv'
end

require 'vin_gvgai'
require 'text'

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    self.x_dim      = args.x_dim
    self.y_dim      = args.y_dim
    self.objects_per_cell = args.objects_per_cell
    self.state_dim  = self.x_dim * self.y_dim * self.objects_per_cell -- State dimensionality.
    -- self.state_dim  = args.state_dim -- State dimensionality.

    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, self.x_dim, self.y_dim}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.text           = {}
    self.text_folder    = args.text_folder
    self.game_ids       = args.game_ids
    self.test_game_ids  = args.test_game_ids
    self.level_ids      = args.level_ids
    self.test_level_ids = args.test_level_ids
    self.current_game   = nil
    self.pretrained_embeddings = args.pretrained_embeddings

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()
    self.expert_network    = args.expert_network

    self.using_vin  = false  -- Set to true if using VIN filters.
    self.vin_k = args.vin_k  -- Number of iterations for VIN.
    self.reactive   = args.reactive
    self.simple   = args.simple
    self.lstm   = args.lstm
    self.vin_only   = args.vin_only
    self.object_filter   = args.object_filter
    self.loss = 0
    self.loss_cnt = 0

    self.debug_flag = 1
    self.debug = false

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end

        -- Load text tensor, etc. 
        self.text = exp.text or {}
        self.word_to_int = exp.word_to_int or {}
        self.word_index = exp.word_index or 500
    
        -- Set the global variables.
        word_to_int = self.word_to_int
        word_index = self.word_index    

    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load expert nets (input is a string containing the dict)
    if self.expert_network then
        loadstring(self.expert_network)()
        -- Now, we have a dictionary named expert_net_list.
        self.expert_network = nil

        for game_num, level_table in pairs(expert_net_list) do 
            for level_num, expert_net in pairs(level_table) do
                local err_msg, exp = pcall(torch.load, expert_net)
                if not err_msg then
                    print("No expert network found.", expert_net)                        
                else
                    self.expert_network = self.expert_network or {}
                    self.expert_network[game_num] = self.expert_network[game_num] or {}
                    self.expert_network[game_num][level_num] = exp.model
                end
            end
        end
    end

    
    -- Load text descriptions from files. 
    for _, id in pairs(self.game_ids) do
        self.text[id] = self.text[id] or {}
        for _, level_id in pairs(self.level_ids) do 
            self.text[id][level_id] = readFile(self.text_folder .. "/" .. id .. "." .. level_id)
        end
    end

    for _, id in pairs(self.test_game_ids) do
        self.text[id] = self.text[id] or {}
        for _, level_id in pairs(self.test_level_ids) do 
            self.text[id][level_id] = readFile(self.text_folder .. "/" .. id .. "." .. level_id)
        end
    end


    -- store in agent variables. 
    self.word_to_int = word_to_int
    self.word_index = word_index
    
    -- Load in pretrained embeddings.
    if self.pretrained_embeddings ~= '' then
      print("Reading pre-trained word embeddings from ", self.pretrained_embeddings)
      local embeddings = readWordVec(self.pretrained_embeddings)

      -- Get only the embeddings we require - initialize model.
      local t = self.network:findModules('nn.LookupTableMaskZero')[1]
      for word, int in pairs(self.word_to_int) do 
        num = tonumber(word)
        if not num and embeddings[word] then  
          t.weight[int] = torch.Tensor(embeddings[word])
        elseif not num then
          print("Embedding not found for ", word)
        end
      end  
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.lastGame = nil
    self.lastLevel = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)
    self.m = self.dw:clone():fill(0)  -- momentum

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end

    return rawstate
end

function nql:filterQ(s, q, backward, batch_size)
    local mask = s:eq(1):float()

    local batch_size = batch_size or self.minibatch_size
    mask = mask:reshape(batch_size, 
                        self.input_dims[1] * self.objects_per_cell,
                        self.input_dims[2], 
                        self.input_dims[3])
    
    mask = mask:sum(2)
    mask = mask:repeatTensor(1, self.n_actions, 1, 1)  -- agent is 1

    q = q:clone():cmul(mask)

    if not backward then        
        q = q:sum(3):sum(4):squeeze()   
    end

    return q
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta, id, level_id
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term
    id = args.id
    level_id = args.level_id
    batch_size = args.batch_size or self.minibatch_size

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local batch_size = a:size(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- create the tensor for text description.

    local text = torch.zeros(batch_size * max_sentences, max_sent_length)

    for i=1, batch_size do 
        text[{{(i-1) * max_sentences + 1, i * max_sentences}, {}}] = self.text[id[i]][level_id[i]]
    end

    if self.gpu >= 0 then
        s = s:cuda()
        s2 = s2:cuda()
        text = text:cuda()
    end

    -- Compute max_a Q(s_2, a).
    q2_max, grad2shape = unpack(target_q_net:forward({s2, text}))
    q2_max = q2_max:float()

    if self.debug and self.debug_flag == 1 then
        print("Q2 values: ", q2_max)
    end


    if self.using_vin then
        q2_max = self:filterQ(s2, q2_max, false, batch_size)
    end

    q2_max = q2_max:max(2)

    
    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end

    delta:add(q2)

    -- q = Q(s,a)
    local q_all
    q_all, _ = unpack(self.network:forward({s, text}))
    q_all = q_all:float()

    if self.using_vin then
        q_all = self:filterQ(s, q_all, false, batch_size)
    end    

    if self.debug and self.debug_flag == 1 then
        print("Q values: ", q_all)
        self.debug_flag = 0
    end

    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max, text, grad2shape:clone():zero()
end


function nql:expertLearnMinibatch()
    -- Perform a minibatch update using expert provided actions.

    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term, id, level_id, expert_actions = self.transitions:sample(self.minibatch_size)

    local batch_size = self.minibatch_size
    local text = torch.zeros(batch_size * max_sentences, max_sent_length)    
    
    -- -- Use the expert network to get Q values. 
    -- q_expert, _ = unpack(self.expert_network:forward({s:float(), text:float()}))

    -- -- convert to prob dist. 
    -- q_expert:exp()
    -- q_expert:cdiv(q_expert:sum(2):expandAs(q_expert))
    -- local expert_actions
    -- _, expert_actions = q_expert:max(2)
    -- expert_actions = expert_actions:squeeze()

    
    for i=1, batch_size do 
        text[{{(i-1) * max_sentences + 1, i * max_sentences}, {}}] = self.text[id[i]][level_id[i]]
    end


    local policy_net = nn.LogSoftMax()
    local criterion = nn.ClassNLLCriterion()

    if self.gpu >= 0 then
        s = s:cuda()
        s2 = s2:cuda()
        text = text:cuda()
        policy_net:cuda()
        criterion:cuda()
        expert_actions = expert_actions:cuda()
        q_expert = q_expert:cuda()
    end

    -- zero gradients.
    policy_net:zeroGradParameters()
    self.dw:zero()


    -- Now train the network using a cross entropy loss
    q, grad2shape = unpack(self.network:forward({s, text}))

    q_policy = policy_net:forward(q)
    local loss = criterion:forward(q_policy, expert_actions)

    -- if self.numSteps % 500 == 0 then
    --     print("states: ", s:sum(2):squeeze())
    --     print("q:", q)
    --     print("policy, expert:", q_policy:clone():exp(), expert_actions)
    --     print("expert_q:", q_expert)
    -- end

    self.loss = self.loss + loss/batch_size
    self.loss_cnt = self.loss_cnt + 1


    local grad = criterion:backward(q_policy, expert_actions)
    grad2 = policy_net:backward(q, grad)
    self.network:backward({s, text}, {-grad2, grad2shape:clone():zero()})

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    -- self.g:mul(0.95):add(0.05, self.dw)
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g2:mul(0.95):add(0.05, self.tmp)
    -- self.tmp:cmul(self.g, self.g)
    -- self.tmp:mul(-1)
    -- self.tmp:add(self.g2)
    -- self.tmp:add(0.01)
    -- self.tmp:sqrt()

    --rmsprop
    -- local smoothing_value = 1e-8
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g:mul(0.9):add(0.1, self.tmp)
    -- self.tmp = torch.sqrt(self.g)
    -- self.tmp:add(smoothing_value)  --negative learning rate

    -- -- accumulate update (for orig and rmsprop)
    -- self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    

    --Adam
    local smoothing_value = 1e-8
    self.tmp:cmul(self.dw, self.dw)
    self.g:mul(0.999):add(0.001, self.tmp)
    self.m:mul(0.9):add(0.1, self.dw)
    self.tmp = torch.sqrt(self.g)
    self.tmp:add(smoothing_value)  --negative learning rate    
    self.deltas:mul(0):addcdiv(self.lr, self.m, self.tmp) -- accumulate update
    
    -- common update
    self.w:add(self.deltas)
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term, id, level_id = self.transitions:sample(self.minibatch_size)

    local targets, delta, q2_max, text, grad2shape = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true, id=id, level_id=level_id}

    -- zero gradients of parameters
    self.dw:zero()

    local batch_size = a:size(1)


    self.network:backward({s, text}, {targets, grad2shape:clone():zero()})

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    -- self.g:mul(0.95):add(0.05, self.dw)
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g2:mul(0.95):add(0.05, self.tmp)
    -- self.tmp:cmul(self.g, self.g)
    -- self.tmp:mul(-1)
    -- self.tmp:add(self.g2)
    -- self.tmp:add(0.01)
    -- self.tmp:sqrt()

    --rmsprop
    -- local smoothing_value = 1e-8
    -- self.tmp:cmul(self.dw, self.dw)
    -- self.g:mul(0.9):add(0.1, self.tmp)
    -- self.tmp = torch.sqrt(self.g)
    -- self.tmp:add(smoothing_value)  --negative learning rate

    -- -- accumulate update (for orig and rmsprop)
    -- self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    

    --Adam
    local smoothing_value = 1e-8
    self.tmp:cmul(self.dw, self.dw)
    self.g:mul(0.999):add(0.001, self.tmp)
    self.m:mul(0.9):add(0.1, self.dw)
    self.tmp = torch.sqrt(self.g)
    self.tmp:add(smoothing_value)  --negative learning rate    
    self.deltas:mul(0):addcdiv(self.lr, self.m, self.tmp) -- accumulate update
    
    -- common update
    self.w:add(self.deltas)
end


function nql:sample_validation_data()
    local s, a, r, s2, term, id, level_id = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
    self.valid_id = id:clone()
    self.valid_level_id = level_id:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max, text = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term,
        batch_size=self.valid_size, id = self.valid_id, level_id = self.valid_level_id}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()

    self.debug_flag = 1
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    -- local state = self:preprocess(rawstate):float()

    local state = rawstate  -- No preprocessing

    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)    

    --Store transition s, a, r, s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, self.lastGame, self.lastLevel, self.lastExpertAction)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, self.objects_per_cell, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(curState, testing_ep)
    end

    self.transitions:add_recent_action(actionIndex)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            if self.expert_network then
                self:expertLearnMinibatch()
            else
                self:qLearnMinibatch()
            end
        end
    end

    if self.expert_network then
        local text = torch.zeros(1 * max_sentences, max_sent_length)    

        local expert_net = self.expert_network[self.current_game][self.current_level]        
    
        -- Use the expert network to get Q values. 
        q_expert, _ = unpack(expert_net:forward({curState:float(), text:float()}))

        -- convert to prob dist. 
        q_expert:exp()
        q_expert:cdiv(q_expert:sum(2):expandAs(q_expert))
        local expert_actions
        _, expert_actions = q_expert:max(2)
        expert_action = expert_actions:squeeze()
    end

    if not testing then
        self.numSteps = self.numSteps + 1
        self.lastState = state:clone()
        self.lastAction = actionIndex
        self.lastTerminal = terminal
        self.lastGame = self.current_game
        self.lastLevel = self.current_level
        self.lastExpertAction = expert_action or 1
    end


    
    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    if not terminal then        
        return actionIndex
    else
        return 0
    end
end


function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    -- Turn single state into minibatch.  Needed for convolutional nets. 
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end
    
    local text = self.text[self.current_game][self.current_level]:float():repeatTensor(1, 1)
    if self.gpu >= 0 then
        state = state:cuda()
        text = text:cuda()
    end

    local q, q_vin = unpack(self.network:forward({state, text}))
    q = q:float():squeeze()

    self.tmp_cnt = self.tmp_cnt or 0
    self.tmp_cnt = self.tmp_cnt + 1

    -- if true or self.tmp_cnt > 1 then        
    --     print("Q_vin: ", q_vin:transpose(3,4))
    --     print("q vin max:", q_vin:max())
    --     print(state:transpose(4,5))        
    --     state_reshaped = state:reshape(1, self.input_dims[1] * self.objects_per_cell, 
    --                self.input_dims[2],
    --                self.input_dims[3])
    --     local q_filtered = nn.ValueFilter(1):forward({state_reshaped, q_vin})
    --     print("Q filtered: ", q_filtered)
    --     print("Q reactive:", self.network.modules[39].output)
    --     print("Q final: ", q)
    --     io.read()
    -- else
    --     print(self.tmp_cnt)
    -- end

    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end
