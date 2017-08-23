main:
	nvcc main.cu -std=c++11 -default-stream per-thread -Wno-deprecated-gpu-targets -O2 -ltbb -Xcompiler -fopenmp -o ca

clean:
	rm -rf ca
