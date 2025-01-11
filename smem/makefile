run:
	nvcc -arch=compute_80 -code=sm_80 test.cu -o test
	cuobjdump -sass test > log
	./test
	ncu --set full --import-source on -f -o log ./test