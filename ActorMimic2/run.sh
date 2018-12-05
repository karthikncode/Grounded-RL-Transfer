# Script to run the server and agent. 
START_PORT=$1
exp_folder=$2

#for i in {0..12}; do 
#for i in {13..28}; do 
game="95"
levels="0,1,2,3,4,5,6"
for i in $(echo $levels | tr "," "\n"); do 
    PORT=$(( $i + $START_PORT));
    cd ../gvgai;
    java -Djava.library.path=/usr/local/lib/ -Dfile.encoding=UTF-8 -classpath "out/production/gvgai:lib/*" Test -p $PORT  >/dev/null &
done;
cd ../ActorMimic2/scripts;
./run_amn_polreg_paper $START_PORT logs/$exp_folder $game $levels
