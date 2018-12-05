# Script to run the server and agent. 
START_PORT=$1
TAGS=$2
target_env=$3
GPU=false


if [ "$target_env" = "fe1" ] ; then
  env_id="95"
  level="0,1,2,3,4,5,6"
elif [ "$target_env" = "fe2" ] ; then
  env_id="96"
  level="7,8,9"
elif [ "$target_env" = "fe2_multitask" ] ; then
  env_id="96"
  level="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
elif [ "$target_env" = "freeway" ] ; then
  env_id="38"
  level="0,1,2,3,4"
elif [ "$target_env" = "boulderchase" ] ; then
  env_id="10"
  level="0,1,2,3,4"
elif [ "$target_env" = "bomberman" ] ; then
  env_id="9"
  level="0,1,2,3,4"
fi;


# Replace the entries below with your own trained models
declare -a models=(
"\"logs/bomberman.jun14_2018.vin.k1.lstm.9.0,1,2,3,4.lr0.0001.eps1.0.seed1.expert.frac1.0.6000/gvgai_6000.t7\""
"\"logs/bomberman.jun14_2018.vin.k1.lstm.9.0,1,2,3,4.lr0.0001.eps1.0.seed2.expert.frac1.0.6001/gvgai_6001.t7\""
"\"logs/bomberman.jun14_2018.vin.k1.lstm.9.0,1,2,3,4.lr0.0001.eps1.0.seed3.expert.frac1.0.6002/gvgai_6002.t7\""
)

# Provide appropriate tagnames for the models above, for logging purposes.
declare -a modelNames=( textvin1_lstm1 textvin1_lstm2 textvin1_lstm3 )

i=0
cnt=0
for (( i=0; i < ${#models[@]}; i++)); do
    model=${models[$i]};
    modelName=${modelNames[$i]};
    train=$env_id
    test=$env_id
    for seed in 1 2 3; do
      for text_fraction in 1.0; do 
        test_level=$level
        for vin_k in 0; do
            for lr in 0.0005; do 
                for bs in 32; do
                  for eps_start in 0.1; do
                      PORT=$((START_PORT + cnt));
                      name=$TAGS.train$train.$level.test$test.$test_level.lr$lr.seed$seed.ep$eps_start.frac$text_fraction.$PORT.model${modelName};
                      cd gvgai;
                      java -Djava.library.path=/usr/local/lib/ -Dfile.encoding=UTF-8 -classpath "out/production/gvgai:lib/*" Test -p $PORT  >/dev/null &
                      cd ../vin;

                      if [ "$GPU" = true ] ; then
                        cuda=$((cnt % 8));
                        CUDA_VISIBLE_DEVICES=$cuda ./run_gpu $PORT logs/$name $train $level $test $test_level $vin_k $lr $bs "nil" $eps_start $model $text_fraction &
                      else
                      ./run_cpu $PORT logs/$name $train $level $test $test_level $vin_k $lr $bs "nil" $eps_start $model $text_fraction &
                      fi;
                      cd ..;
                      cnt=$((cnt + 1));
                  done;
                done;             
            done;
        done;
      done;
    done; 
done;

