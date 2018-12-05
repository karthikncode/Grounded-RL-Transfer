
local env = torch.class('GameEnv')

local json = require ("dkjson")
local zmq = require "lzmq"

if pcall(require, 'signal') then
    signal.signal("SIGPIPE", function() print("raised") end)
else
    print("No signal module found. Assuming SIGPIPE is okay.")
end

INITIAL_PORT = 6000
C_MAX = 2  -- max number of objects per cell
X_MAX = 16
Y_MAX = 16

function env:__init(args)

    self.ctx = zmq.context()
    self.skt = self.ctx:socket{zmq.REQ,
        linger = 0, rcvtimeo = 10000;
        connect = "tcp://127.0.0.1:" .. args.zmq_port;
    }
    self.actions = {}  -- Actions in the game.
    -- for a = 0, args.num_actions-1 do
    --     self.actions[a+1] = a
    -- end


end

function env:process_msg(msg)    
    -- screen, reward, terminal
    -- print("MESSAGE:", msg)
    loadstring(msg)()

    -- print(torch.Tensor(state):transpose(1,2))
    -- _ = io.read()

    -- if reward ~= 0 then
    --     print('non-zero reward', reward)
    -- end    
    rawstate = torch.Tensor(state)
    -- print('state size is : ', rawstate:size())
    c = rawstate:size(1)
    x = rawstate:size(2)
    y = rawstate:size(3)
    zt = torch.zeros(C_MAX,X_MAX,Y_MAX)
    zt[{{1,c},{1,x},{1,y}}] = rawstate
    return zt, reward, terminal
    -- return rawstate, reward, terminal
end

function env:newGame(game_num, level_num)
    self.skt:send("newGame " .. game_num .. " " .. level_num)
    
    msg = self.skt:recv()
    while msg == nil do
        msg = self.skt:recv()
    end

    -- Get the number of actions.
    local num_actions = tonumber(msg)
    if #self.actions == 0 then
        for a = 0, num_actions-1 do
            self.actions[a+1] = a
        end
    end

    -- Send ACK. 
    self.skt:send("ACK")
    msg = self.skt:recv()
    while msg == nil do
        msg = self.skt:recv()
    end

    return self:process_msg(msg)
end

function env:step(action)    
    -- print("Sending action ", action)
    self.skt:send(tostring(action))
    msg = self.skt:recv()
    while msg == nil do
        msg = self.skt:recv()
    end
    return self:process_msg(msg)
end

function env:evalStart()
    self.skt:send("evalStart")
    msg = self.skt:recv()
    assert(msg == 'done', msg)
end

function env:evalEnd()
    self.skt:send("evalEnd")
    msg = self.skt:recv()
    assert(msg == 'done', msg)
end

function env:getActions()   
    return self.actions
end

-- -- Return a random action from the provided ones. 
-- function RandomAction(actions)
--     return actions[math.random(#actions)]
-- end

-- Simple test to make sure ZMQ works. 
-- Comment this out while using as a library.
-- local gameEnv = GameEnv(nil)  -- Args are nil for now, but can be passed in.
-- local gameActions = gameEnv:getActions() 
-- local state, reward, terminal
-- state, reward, terminal = gameEnv:newGame()
-- while true do
--    action = RandomAction(gameActions)
--    state, reward, terminal = gameEnv:step(action)
--    if terminal then
--        print("Starting new game...")
--        state, reward, terminal  = gameEnv:newGame()
--    end
-- end