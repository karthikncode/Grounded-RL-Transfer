--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'networks.convnet'

return function(args)
    args.n_units        = {16, 32, 32}
    args.filter_size    = {4, 2, 1}
    args.filter_stride  = {3, 2, 1}
    args.n_hid          = {128}
    args.nl             = nn.Rectifier

    return create_network(args)
end

