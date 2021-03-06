#!/bin/bash
[ "$1" == "-h" -o "$1" == "-help" ] && echo "
Run the script with following arguments:

a. <path_of_virtual_environment> : This is an optional argument that specifies the path of the virtual environment

Run the script as:

baseline_generator.sh <path_of_virtual_environment>
" && exit

HOME=$(pwd)

# Locating the virtual environment
if [ -z "$1" ]
  then
    virtual=/home/bhattacharya/personal_work_troja/MML/virtual
    [ -d $virtual ] && source $virtual/bin/activate    
else
    virtual=$1
    source $virtual/bin/activate
fi

rm computed/tfidf_baselines.tsv
touch computed/tfidf_baselines.tsv

# AMBIGUITY CASES
AMB_LOC=$HOME/Representations/Ambiguity
./src/tsne.py -d $AMB_LOC/COCO -m "Ambiguity" 
./src/tsne.py -d $AMB_LOC/EMMT -m "Ambiguity" 

# GRAMATICALITY CASES
GRAM_LOC=$HOME/Representations/Grammaticality
for d in $GRAM_LOC/*; do
    n_loc=$d
    bert_repr=$n_loc/BERT.pkl
    if [ -f "$bert_repr" ]; then
        ./src/tsne.py -d $n_loc -m "Grammaticality" 
    else
        for case in $n_loc/*/; do
            # echo $case
            ./src/tsne.py -d $case -m "Grammaticality" 
        done

    fi
done

#COMPLEXITY CASE
COM_LOC=$HOME/Representations/Complexity
./src/tsne.py -d $COM_LOC/complexity_english -m "Complexity" 
./src/tsne.py -d $COM_LOC/complexity_italian -m "Complexity" 