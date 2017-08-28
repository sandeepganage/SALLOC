main: driver.cu salloc.h
	nvcc driver.cu -o driver -std=c++11 -g -G

clean:
	rm -rf driver	
