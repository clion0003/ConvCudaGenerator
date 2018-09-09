#include <iostream>
#include <string>
#include <fstream>
#include <ConvCudaGenerator.h>
#include <mstch/mstch.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include "caffe.pb.h"
using namespace caffe;
using namespace std;

using google::protobuf::io::FileInputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
    int fd = open(filename, O_RDONLY);
    FileInputStream* input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
}

/*parse caffe prototxt and generate kernel code using template engine of mustache*/

int main(int argc, char** argv)
{
    if (argc != 3){
        cout << "error: argc = " << argc << endl;
        cout << "example: ./ConvCudaGenerator conv.cu vgg16.prototxt" <<endl;
        return -1;
    }

    fstream ofile;
    ofile.open(argv[1], ios::out);
    if (!ofile) {
        cout << "can't open " << argv[1] << endl;
    }
    /*parse the prototxt*/
    NetParameter CNN;

    if(!ReadProtoFromTextFile(argv[2], &CNN))
    {
        cerr<<"error opening file"<<endl;
        return -1;
    }

    /*print the header code*/
    ofile << BeginningCode;

    /*print the kernel code for each conv layer*/
    const int layer_size = CNN.layer_size();
    for (int i = 0; i < layer_size; i++) {
        string layer_type = CNN.layer(i).type();
        if (!strcmp(layer_type.c_str(), "Convolution")){
            mstch::map context_kernel{
                    {"info_kernel",
                            mstch::array
                                    {
                                            mstch::map{{"name", string{CNN.layer(i).name()}}
                                            }
                                    }
                    }
            };
            ofile << mstch::render(KernelCode, context_kernel);
        }
    }

    /*print the conv code calling kernels*/
    ofile << ConvCodeHead << endl;
    for (int i = 0;i < layer_size; i++) {
        string layer_type = CNN.layer(i).type();
        if (!strcmp(layer_type.c_str(), "Convolution")){
            mstch::map context_conv{
                    {"info_conv",
                            mstch::array
                                    {
                                            mstch::map{{"name", string{CNN.layer(i).name()}}
                                            }
                                    }
                    }
            };
            ofile << mstch::render(ConvCode, context_conv);
        }
    }

    /*print the ending code*/
    ofile<<EndingCode;
    ofile.close();

    /*make libconv.so*/
    system(("nvcc -ccbin g++-5 --cudart=shared -Xcompiler -fPIC -shared -o libaigpu.so " + string(argv[1])).c_str());
}
