# Script to run the server and agent (with optional expert model for teacher forcing)
# Different modes are -- dqn, textdqn, vin, textvin
START_PORT=$1
env=$2  # Choose from ["fe1", "fe2", "freeway", "boulderchase", "bomberman"]
TAGS=$3
GPU=false

TAGS=$env.$TAGS

if [ "$env" = "fe1" ] ; then
  env_id="95"
  level="0,1,2,3,4,5,6"
elif [ "$env" = "fe2" ] ; then
  env_id="96"
  level="7,8,9"
elif [ "$env" = "fe2_multitask" ] ; then
  env_id="96"
  level="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
elif [ "$env" = "freeway" ] ; then
  env_id="38"
  level="0,1,2,3,4"
elif [ "$env" = "boulderchase" ] ; then
  env_id="10"
  level="0,1,2,3,4"
elif [ "$env" = "bomberman" ] ; then
  env_id="9"
  level="0,1,2,3,4"
fi;

# Use this option if you want to use teacher forcing with another pre-trained expert model.
#expert_model="\"logs/16x16_reactive.95.4.expert.6020/gvgai_6020.t7\""
expert_model="nil"

cnt=0

# You can run different models by changing the mode to a value in ['dqn', 'vin', 'textdqn', 'textvin'].
for mode in "textvin"; do                
  for textrep in "lstm"; do
    if [ "$mode" = "vin" ] || [ "$mode" = "textvin" ]; then
      vin_k_values=( 1 3 )
    else
      vin_k_values=( 0 )
    fi;

    for vin_k in "${vin_k_values[@]}"; do
      for text_fraction in 1.0; do 
        for lr in 0.0005; do
          for eps_start in 1.0; do
            for seed in 1; do  #just a dummy - seed=PORT
              PORT=$((START_PORT + cnt));
              name=$TAGS.$mode.k$vin_k.$textrep.$env_id.$level.lr$lr.eps$eps_start.seed$seed.expert.frac$text_fraction.$PORT;
              cd gvgai;
              java -Djava.library.path=/usr/local/lib/ -Dfile.encoding=UTF-8 -classpath "out/production/gvgai:lib/*" Test -p $PORT  >/dev/null &
              cd ../vin;
              
              if [ "$GPU" = true ] ; then
                cuda=$((cnt % 8));
                CUDA_VISIBLE_DEVICES=$cuda ./run_gpu $PORT logs/$name $env_id $level $env_id $level $vin_k $lr 32 $expert_model $eps_start &
              else
                ./run_cpu $PORT logs/$name $env_id $level $env_id $level $vin_k $lr 32 $expert_model $eps_start "\"vin_gvgai\"" $text_fraction $mode $textrep &
              fi
              

              
              cd ..;
              cnt=$((cnt + 1));
            done;
          done;
        done;
      done;
    done;
  done;
done;



