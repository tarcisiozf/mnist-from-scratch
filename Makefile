CCFLAGS = -O3 -Wall -std=c2x
FILES=matrix.c engine.c mem.c params.c batch.c dataset.c mmap.c bench.c

default: train test

train: build-cpu
	./train.bin

build-cpu:
	gcc $(CCFLAGS) -o train.bin $(FILES) main.c -lm

cuda:
	nvcc -c -Xcompiler -fPIC cuda.cu -o cuda.o && \
	nvcc -shared -o libmnist_cuda.so cuda.o && \
	gcc -DCUDA -I. *.c -o train.bin -lm \
		-L. -lmnist_cuda \
		-L/usr/local/cuda/lib64 -lcudart \
		-Wl,-rpath='$ORIGIN:./:/usr/local/cuda/lib64'

clean:
	rm -rf *.o *.so *.bin
