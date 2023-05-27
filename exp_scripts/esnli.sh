#ESNLi
NUM_TR=512
NUM_DEV=500

random_shots_exp()
{
    CODEXV=$1
    SLICE_TR=$2
    if [ ! -z "$SLICE_TR" -a ! -z "$SLICE_TR" ]
    then
        OPENAI_API_KEY=${KEY} python run_selective.py --task esnli --do_random_scoring --num_dev ${NUM_DEV} --eng code-davinci-${CODEXV} --run_pred --slice_train ${SLICE_TR} --num_train ${NUM_TR} --batch_size 20 --style_template psource
    fi
}

bertscore_exp()
{
    CODEXV=$1
    SLICE_TR=$2
    if [ ! -z "$SLICE_TR" -a ! -z "$SLICE_TR" ]
    then
        OPENAI_API_KEY=${KEY} python run_selective.py --task esnli --num_dev ${NUM_DEV} --eng code-davinci-${CODEXV}  --run_pred --slice_train ${SLICE_TR} --num_train ${NUM_TR} --score_strategy esnli-promptquestion --style_template psource --do_bertscore --selection_strategy nn --batch_size 20
        OPENAI_API_KEY=${KEY} python run_selective.py --task esnli --num_dev ${NUM_DEV} --eng code-davinci-${CODEXV}  --run_pred --slice_train ${SLICE_TR} --num_train ${NUM_TR} --score_strategy esnli-promptquestion --style_template psource --do_bertscore --selection_strategy mmr  --batch_size 20
    fi
}


random_shots_exp 002 0 # other random trial 1024, 2048, 3072
bertscore_exp 002 0 # other random trial 1024, 2048, 3072
