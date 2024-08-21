# MNIST FROM SCRATCH

This is a simple implementation of a neural network to classify the MNIST dataset. 
The neural network is implemented from scratch. 

This a study experiment to understand the basics of neural networks and backpropagation. 
Not having numpy or any other library forced me to implement my own lib (see [matrix.cpp](./matrix.cpp)), with element-wise operations, broadcasting and so on.

## Neural network

The neural network has 3 layers: input, hidden and output. with layout 784-800-10. That's enough to reach 92% accuracy on the dataset.

The repo contains a subset of MNIST dataset with 32k samples, 1k used for validation and the rest for training.

## Compiling

### CPU version
In case you don't have a GPU, you can train the neural network using the CPU:
```bash
make cpu
```

### GPU version
If you have CUDA hardware, nvcc (CUDA Toolkit) is required:
```bash
make cuda
```
Just by using a naive implementing of the matrix dot operation with CUDA I got a 28x speedup.
The idea is not use libraries like cuBLAS, and so on, but to implement the operations from scratch.

## Running

After compiling, you can run the neural network:
```bash
./train.bin
```

It will run gradient descent with 1000 iterations and print the accuracy of the neural network using sampled test data.

## Notes
* This is a very simple implementation, not production level quality.
* I'm no C/CUDA expert, so it may not be as fluent.

## Lessons learned

The code was based on a Python tutorial, and I wanted to port to another language from scratch. Go was the first option because of its simplicity and performance.
And the second step was to port to C++ with CUDA to leverage the GPU.

After porting the code from Python to Go, I didn't get the expected results, my gradient descent was getting worse and worse.

To fix that first I had to have deterministic results for both implementations, so I pre-trained on python and dumped the parameters in JSON, so I could reuse them on Go. Then I did debugging on both languages, parameter by parameter looking for bugs

That didn't work at scale, so I had to build a tool to compare the outputs of both implementations, so I created what I called "memstate" in both, which worked similarly to a memory dump, so I could compare the state of the network at any given time

I choose a third language (JS) to rule possible issues floating-point implementations and slowly started increasing the size of inputs to see the differences and once spotted I resorted back to manual debugging

Finally I was able to catch a few bugs, one of them was my softmax implementation, that was built by porting `np.sum`, not python's builtin `sum` and that was the root cause :)