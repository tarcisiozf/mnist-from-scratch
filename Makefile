cpu:
	g++ -o train.bin main.cpp matrix.cpp engine.cpp -lm

cuda:
	nvcc -DCUDA -o train.bin main.cpp cuda.cu engine.cpp