

local env = torch.class('GameEnv')

local json = require ("dkjson")
local zmq = require "lzmq"

if pcall(require, 'signal') then
    signal.signal("SIGPIPE", function() print("raised") end)
else
    print("No signal module found. Assuming SIGPIPE is okay.")
end

ZMQ_PORT = 6000

function env:__init(args)

    self.ctx = zmq.context()
    self.skt = self.ctx:socket{zmq.REQ,
        linger = 0, rcvtimeo = 10000;
        connect = "tcp://127.0.0.1:" .. ZMQ_PORT;
    }
    self.actions = {0,1,2,3}  -- Actions in the game.
end

function env:process_msg(msg)    
    -- screen, reward, terminal
    print("MESSAGE:", msg)
    loadstring(msg)()
    -- if reward ~= 0 then
    --     print('non-zero reward', reward)
    -- end    
    return torch.Tensor(state), reward, terminal
end

function env:newGame()
    self.skt:send("newGame")
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

-- Return a random action from the provided ones. 
function RandomAction(actions)
    return actions[math.random(#actions)]
end

-- Simple test to make sure ZMQ works. 
-- Comment this out while using as a library.
local gameEnv = GameEnv(nil)  -- Args are nil for now, but can be passed in.
local gameActions = gameEnv:getActions() 
local state, reward, terminal
state, reward, terminal = gameEnv:newGame()
while true do
    action = RandomAction(gameActions)
    state, reward, terminal = gameEnv:step(action)
    if terminal then
        print("Starting new game...")
        state, reward, terminal  = gameEnv:newGame()
    end
end