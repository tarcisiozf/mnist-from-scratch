cpu:
	gcc -o train.bin *.c -lm

cuda:
	nvcc -c -Xcompiler -fPIC cuda.cu -o cuda.o && \
	nvcc -shared -o libmnist_cuda.so cuda.o && \
	gcc -DCUDA -I. *.c -o train.bin -lm \
		-L. -lmnist_cuda \
		-L/usr/local/cuda/lib64 -lcudart \
		-Wl,-rpath='$ORIGIN:./:/usr/local/cuda/lib64'

clean:
	rm -rf *.o *.so *.bin
