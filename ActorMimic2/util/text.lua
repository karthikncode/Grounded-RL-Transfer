-- functions to read text descriptions and handle them. 

word_to_int = {}
word_index = 500  -- allow first 500 to represent object ids in state.

max_sent_length = 6
max_vocab = 100
max_sentences = 6

text_tensor = torch.zeros(max_sentences, max_sent_length)

function readFile(filename)
    io.input(filename)
    local sent_num = 1
    for line in io.lines() do                 
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
            text_tensor[sent_num][i] = word_to_int[word]
        end
        sent_num = sent_num + 1
    end
    return text_tensor
end
