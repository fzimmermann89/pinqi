#MSUB -S /bin/bash
#MSUB -l nodes=1:ppn=16:gpus=1,walltime=72:00:00
#MSUB -l feature=epyc7713
#MSUB -q gpu

PYTHON=python
CONFIGS=(configs/config-sat2.yaml configs/ds_Nc32Ac4.yaml configs/data_sat.yaml)
DATAPATH=data/brainwebClasses/
PATH=.
join() {
  local separator="$1"
  shift
  printf "%s" "${@/#/$separator}"
}
CONFIGSTR=$(join ' -c ' ${CONFIGS[@]})
export NEPTUNE_MODE='offline'

cd $PATH && $PYTHON train.py  $CONFIGSTR  --data.init_args.path=$DATAPATH
