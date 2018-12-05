-- functions to read text descriptions and handle them. 

require 'utils'

local _ = require 'underscore'

word_to_int = {}
word_index = 500  -- allow first 500 to represent object ids in state.

max_sent_length = 25
max_vocab = 300
max_sentences = 26

text_tensor = torch.zeros(max_sentences, max_sent_length)

print("TEXT_FRACTION:", TEXT_FRACTION)

function readFile(filename)
    io.input(filename)
    local sent_num = 1
    for line in io.lines() do                 
        j = 1
        for i, word in pairs(split(line, "%S+")) do
            if not word_to_int[word] then
                num = tonumber(word)
                if num then
                    word_to_int[word] = num
                else
                    word_to_int[word] = word_index
                    word_index = word_index + 1
                end
            end
            -- selectively insert words into description input.
            if ((i < 2) or (torch.uniform() <= TEXT_FRACTION))  then
                text_tensor[sent_num][j] = word_to_int[word]
                j = j + 1
            end
        end
        sent_num = sent_num + 1
    end
    return text_tensor:clone()
end

-- read in word vectors - one per line
function readWordVec(filename)
    local file = io.open(filename, "r");
    local data = {}
    local parts
    local wordVec = {} -- global
    for line in file:lines() do
        parts = line:split(" ")
        wordVec[parts[1]] = _.rest(parts)
    end
    return wordVec
end
