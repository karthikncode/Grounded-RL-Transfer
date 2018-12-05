local ValueFilter, parent = torch.class('nn.ValueFilter', 'nn.Module')

-- This module accepts minibatches
-- Input is of form {s, q}

function ValueFilter:__init(agent_id)
    self.agent_id = agent_id
end


function ValueFilter:updateOutput(input)
    -- return self.output:resizeAs(input):copy(input):abs():add(input):div(2)

    -- function nql:filterQ(s, q, backward, batch_size)

    local mask = input[1]:eq(self.agent_id):typeAs(input[1])
    mask = mask:sum(2)

    mask = mask:repeatTensor(1, input[2]:size(2), 1, 1)  -- agent is 1


    self.output = input[2]:clone():cmul(mask)

    self.output = self.output:sum(3):sum(4):squeeze(4):squeeze(3)       
    
    return self.output
end

function ValueFilter:updateGradInput(input, gradOutput)
    local mask = input[1]:eq(self.agent_id):typeAs(input[1])
    mask = mask:sum(2)
    mask = mask:repeatTensor(1, input[2]:size(2), 1, 1)  -- agent is 1

    self.gradInput = {input[1]:clone():zero(), 
                      gradOutput:repeatTensor(input[1]:size(3),
                                              input[1]:size(4), 
                                              1, 1)
                                :transpose(1, 3):transpose(2, 4)
                                :cmul(mask),
                      0}
    
    return self.gradInput
end
