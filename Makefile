main: driver_phase3.cu salloc_phase3.h
	nvcc driver_phase3.cu -o driver_phase3 -g -G

clean:
	rm -rf driver_phase3	
