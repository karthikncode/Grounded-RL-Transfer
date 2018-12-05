
require 'networks.vin'

return function(args)
    args.r_channels = {1}
    args.r_filter = {3, 3}
    args.r_filter_stride = {1, 1}

    args.p_filter = {3, 3}
    args.p_filter_stride = {1, 1}

    args.q_filter = {3, 3}
    args.q_filter_stride = {1, 1}    

    -- For reactive component.
    -- args.n_units        = {32, 64, 64}
    -- args.filter_size    = {8, 4, 3}
    -- args.filter_stride  = {4, 2, 1}
    -- args.n_hid          = {256, 128}
    args.n_units        = {16, 32}
    args.filter_size    = {4, 2}
    args.filter_stride  = {3, 2}
    args.n_hid          = {128}
    args.nl             = nn.ReLU

    return create_network(args)
end

