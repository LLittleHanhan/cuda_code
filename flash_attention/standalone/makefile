run:
	nvcc tiny_flash_attention_v3.cu -I /root/byf/opensource/cutlass/include -o test -std=c++17 -arch=compute_80 -code=compute_80,sm_80 --expt-relaxed-constexpr
	# nvcc reference.cu -I /root/byf/opensource/cutlass/include -o test -std=c++17 -arch=compute_80 -code=compute_80,sm_80 --expt-relaxed-constexpr
	cuobjdump test -sass > ./log/log
	ncu --set full -f --import-source on -o ./log/log ./test
	./test