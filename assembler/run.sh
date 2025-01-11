export PATH=${HOME}/hky/python3/bin:$PATH
if [[ ":$PATH:" != *":./CuAssembler/bin:"* ]]; then
    export PATH=${PATH}:./CuAssembler/bin
fi
if [[ ":$PYTHONPATH:" != *":./CuAssembler:"* ]]; then
    export PYTHONPATH=${PYTHONPATH}:./CuAssembler
fi

rm ./CuAssembler/bin/cuasm
ln -s cuasm.py ./CuAssembler/bin/cuasm
chmod a+x ./CuAssembler/bin/cuasm

rm test
rm ./temp/*
nvcc -arch=compute_70 -code=compute_70,sm_70 -lineinfo -keep -keep-dir ./temp test.cu -o test
ncu --set full --import-source on -f -o ./result/origin ./test
nvcc -arch=compute_70 -code=compute_70,sm_70 -lineinfo -dryrun -keep -keep-dir ./temp test.cu -o test 2> compile.sh
python3 modify_compile.py

rm test.sm_70.cuasm
mv ./temp/test.sm_70.cubin ./
cuasm test.sm_70.cubin

## modify
python3 modify.py


rm test.sm_70.cubin
cuasm test.sm_70.cuasm
mv test.sm_70.cubin ./temp

rm test
sh compile.sh
chmod a+x compile.sh
ncu --set full --import-source on -f -o ./result/now ./test

