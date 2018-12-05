# Script to run the server and agent. 
START_PORT=$1
exp_folder=$2
transfer=$3
seed=$4
game="96"
levels="7,8,9" 
for i in $(echo $levels | tr "," "\n"); do 
    PORT=$(( $i + $START_PORT));
    cd ../gvgai;
    java -Djava.library.path=/usr/local/lib/ -Dfile.encoding=UTF-8 -classpath "/scratch/karthikn/Grounded-transfer/gvgai/out/production/gvgai:lib/*" Test -p $PORT >/dev/null &
done;
cd ../ActorMimic2/scripts;
./run_dqn_polreg_paper_gvgai $START_PORT logs/$exp_folder $game $levels $transfer $seed

