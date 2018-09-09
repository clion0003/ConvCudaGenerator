.PHONY : clean
LibAIGPUGen: src/main.cpp src/caffe.pb.cc
	g++ src/main.cpp src/caffe.pb.cc -I./include -I$(HOME)/local/include -std=c++11 -Wall -O2 -L$(HOME)/local/lib -lprotobuf -lmstch -o LibAIGPUGen.bin
clean:
	-rm *.bin
