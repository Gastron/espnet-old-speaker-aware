module purge
module load kaldi-vanilla
module unload CUDA
module load protobuf cmake anaconda3
module list
source activate espnet-py3

#CUDAROOT=$(which nvcc | xargs dirname | xargs -I "{}" realpath "{}/..")
#export PATH=$CUDAROOT/bin:$PATH
#export LD_LIBRARY_PATH=$CUDAROOT/lib64:$CUDAROOT/lib64/stubs:$LD_LIBRARY_PATH
#export CUDA_HOME=$CUDAROOT
#export CUDA_PATH=$CUDAROOT
#export LDFLAGS="-L$CUDAROOT/lib64 -L$CUDAROOT/lib64/stubs $LDFLAGS"
#export CUDA_TOOLKIT_ROOT_DIR=$CUDAROOT

MAIN_ROOT="$PWD/../../.."

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build:$MAIN_ROOT/tools
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH}:$MAIN_ROOT:/scratch/work/rouhea1/espnet-py3/tools/kaldi-io-for-python"
