-- THis module takes in the state and some object embeddings and
-- replaces them into the corresponding id locations in the state. 

local ObjectDescriptor, parent = torch.class('nn.ObjectDescriptor', 'nn.Module')

-- This module accepts minibatches
-- Input is of form {s, q}

function ObjectDescriptor:__init(dim, limit)    
    self.dim = dim  -- dimension of embeddings
    self.limit = limit
end


function ObjectDescriptor:updateOutput(input)
    
    -- input is of form {state, emb_state, obj_ids, emb_objects}

    state, emb_state, obj_ids, emb_objects = unpack(input)
    emb_state = emb_state:clone():zero()

    local mask
    local updated = {}

    -- TODO: this can be optimized.
    for i=1, math.min(self.limit, obj_ids:size(1)) do 
        if not updated[obj_ids[i]] then 
            updated[obj_ids[i]] = true

            mask = state:eq(obj_ids[i]):typeAs(state)
            mask = mask:repeatTensor(self.dim, 1, 1):transpose(1,2):transpose(2,3)
            obj_emb = emb_objects[i]:repeatTensor(state:size(1), state:size(2), 1)

            -- print("MASK:", state, emb_state, obj_ids[i], mask)
            -- io.read()
            -- emb_state:cmul(1 - mask)
            emb_state = emb_state + obj_emb:cmul(mask)        
        end
    end
    -- print(emb_state)
    -- io.read()
    self.output = emb_state
    
    return self.output
end

function ObjectDescriptor:updateGradInput(input, gradOutput)
    -- input is of form {state, emb_state, emb_objects}

    state, emb_state, obj_ids, emb_objects = unpack(input)
    emb_state = emb_state:clone():zero()

    -- local grad_emb_state = gradOutput:clone()
    local grad_emb_objects = emb_objects:clone():zero()

    local mask
    local updated = {}

    -- TODO: make sure gradients go to the right places.
    for i=1, math.min(self.limit, obj_ids:size(1)) do                 
        mask = state:eq(obj_ids[i]):typeAs(state)
        mask = mask:repeatTensor(self.dim, 1, 1):transpose(1,2):transpose(2,3)

        local grad_obj_emb = gradOutput:clone():cmul(mask):mean(1):mean(2)
        -- if not updated[obj_ids[i]] then
        --     updated[obj_ids[i]] = true
        --     grad_emb_state:cmul(1-mask)  -- zero out the grads for these objects here.
        -- end
        grad_emb_objects[i] = grad_obj_emb
    end    

    if self.limit < obj_ids:size(1) then
        grad_emb_objects = grad_emb_objects[{{1, self.limit}, {}}]:repeatTensor(obj_ids:size(1)/self.limit, 1)
    end

    self.gradInput = {state:clone():zero(), 
                      gradOutput:clone():zero(),   -- gradient only for RNN part
                      obj_ids:clone():zero(),
                      grad_emb_objects}
    
    return self.gradInput
end
