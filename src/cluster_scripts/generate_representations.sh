#!/bin/bash

HOME=$(pwd)

# HUGGINGFACE=$HOME/huggingface_cache
# if  [ ! -d $HUGGINGFACE ]
# then
#     mkdir $HOME/huggingface_cache
# else
#     rm -rf $HOME/huggingface_cache
#     mkdir $HOME/huggingface_cache
# fi


data=$HOME/data
[ ! -d $data ] && echo "Data Directory $data DOES NOT exist. Terminating" && exit

virtual=/home/bhattacharya/personal_work_troja/MML/virtual
[ -d $virtual ] && source $virtual/bin/activate    

python3 $HOME/src/source/representation_generator.py --data $data

# rm -rf $HOME/.cache
# rm -rf $HOME/huggingface_cache
