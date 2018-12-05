# Grounded-transfer

This repository contains the code for our paper [Grounding Language for Transfer in Deep Reinforcement Learning](https://arxiv.org/abs/1708.00133), appearing in the Journal of Artificial Intelligence (JAIR).

### Installation and Setup
You will first need to install [Torch](http://torch.ch/docs/getting-started.html).   

We have included some `signal` library binaries to deal with SIGPIPE issues in Linux. If they don't work out of the box for you, you will need to install the Lua dev library `liblua` (`sudo apt-get install liblua5.2`) and the [signal](https://github.com/LuaDist/lua-signal) package for Torch.
(You may need to uninstall the [signal-fft](https://github.com/soumith/torch-signal) package or rename it to avoid conflicts.)
Install some required Torch packages:
`luarocks install visdom rnn nninit lzmq underscore`

#### Other requirements  
Make sure to install [ZMQ](http://zeromq.org/intro:get-the-software) (`sudo apt-get install libzmq-dev`) and the java port of the library [jzmq](https://github.com/zeromq/jzmq) (this can also be done directly using Maven). 
Also, make sure to set your `java.library.path` to the place containing the zmq and jzmq `.so` files (this is usually at `/usr/local/lib`). This code has been tested to work with Java 1.8.0_171.

### Testing ZMQ (optional)

First run the game server either through the command line or the java IDE.   
Then, run the torch client under the folder `torch-zmq-test` as `th game_env.lua`.  
The client should connect and you will see the game screen pop up.  

### Running the base model for an environment
You can use the script `run_expert.sh` to train a base model on one of the game environments. The script takes three arguments for the ZMQ port number, environment name, and a tag for the run. Here is some sample usage:
```bash
./run_expert.sh 6000 fe1 testrun
```
You can choose to enable GPU training by setting `GPU=true` in `run_expert.sh`.

### Transfer 
Use the script `run_transfer_test.sh` to run transfer experiments. Plug in the path to your pre-trained model(s) in the `models` variable like this:
```bash
declare -a models=(
"\"logs/fe1_testrun.vin.k1.lstm.95.0,1,2,3,4,5,6.lr0.0001.eps1.0.seed2.expert.frac1.0.6000/gvgai_6000.t7\""
"\"logs/fe1_testrun.vin.k2.lstm.95.0,1,2,3,4,5,6.lr0.0001.eps1.0.seed2.expert.frac1.0.6001/gvgai_6001.t7\""
"\"logs/fe1_testrun.vin.k3.lstm.95.0,1,2,3,4,5,6.lr0.0001.eps1.0.seed2.expert.frac1.0.6002/gvgai_6002.t7\""
"\"logs/fe1_testrun.vin.k5.lstm.95.0,1,2,3,4,5,6.lr0.0001.eps1.0.seed2.expert.frac1.0.6003/gvgai_6003.t7\""
)
```
Also, declare model names (used for logging results) for the corresponding files like this:
```bash
declare -a modelNames=( textonly_vin_k1 textonly_vin_k2 textonly_vin_k3 textonly_vin_k5)
```

### Stopping runs
The scripts `run_expert.sh` and `run_transfer_test.sh` launch processes in the background. In order to kill them, you can use:
```bash
killall luajit & killall java
```
