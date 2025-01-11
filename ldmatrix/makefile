run:
	nvcc test.cu -o test -arch=compute_80 -code=sm_80
	./test
	cuobjdump -sass ./test > log
	ncu --set full --import-source on -f -o log ./test
