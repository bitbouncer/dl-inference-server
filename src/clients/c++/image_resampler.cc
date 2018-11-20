#include "src/clients/c++/request.h"

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs.hpp>
#include "src/core/model_config.pb.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

namespace {

enum ProtocolType {
  HTTP = 0,
  GRPC = 1
};

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr
    << "Usage: " << argv[0] << " [options] <image filename / image folder>"
    << std::endl;
  std::cerr
    << "    Note that image folder should only contain image files."
    << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t-c <topk>" << std::endl;
  std::cerr << "\t-s <NONE|INCEPTION|VGG>" << std::endl;
  std::cerr << "\t-p <proprocessed output filename>" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-x <model version>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
    << std::endl;
  std::cerr << std::endl;
  std::cerr
    << "For -b, the image will be replicated and sent in a batch" << std::endl
    << "        of the specified size. Default is 1." << std::endl;
  std::cerr
    << "For -c, the <topk> classes will be returned, default is 1."
    << std::endl;
  std::cerr
    << "For -s, specify the type of pre-processing scaling that" << std::endl
    << "        should be performed on the image, default is NONE." << std::endl
    << "    INCEPTION: scale each pixel RGB value to [-1.0, 1.0)." << std::endl
    << "    VGG: subtract mean BGR value (104, 117, 123) from"
    << std::endl << "         each pixel." << std::endl;
  std::cerr
    << "If -x is not specified the most recent version (that is, the highest "
    << "numbered version) of the model will be used." << std::endl;
  std::cerr
    << "For -p, it generates file only if image file is specified."
    << std::endl;
  std::cerr
    << "For -u, the default server URL is localhost:8000." << std::endl;
  std::cerr
    << "For -i, available protocols are gRPC and HTTP. Default is HTTP."
    << std::endl;
  std::cerr << std::endl;

  exit(1);
}

ProtocolType
ParseProtocol(const std::string& str)
{
  std::string protocol(str);
  std::transform(
    protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
  if (protocol == "http") {
    return ProtocolType::HTTP;
  } else if (protocol == "grpc") {
    return ProtocolType::GRPC;
  }

  std::cerr
    << "unexpected protocol type \"" << str
    << "\", expecting HTTP or gRPC" << std::endl;
  exit(1);

  return ProtocolType::HTTP;
}

void
ParseModel(
  const std::unique_ptr<nic::InferContext>& ctx, const size_t batch_size, size_t* c, size_t* h, size_t* w, ni::ModelInput::Format* format, bool verbose = false)
{
  if (ctx->Inputs().size() != 1) {
    std::cerr
      << "expecting 1 input, model \"" << ctx->ModelName() << "\" has "
      << ctx->Inputs().size() << std::endl;
    exit(1);
  }

  if (ctx->Outputs().size() != 1) {
    std::cerr
      << "expecting 1 output, model \"" << ctx->ModelName() << "\" has "
      << ctx->Outputs().size() << std::endl;
    exit(1);
  }

  const auto& input = ctx->Inputs()[0];

  *format = input->Format();

  if (input->Dims().size() != 3) {
    std::cerr
      << "expecting model input to have 3 dimensions, model \""
      << ctx->ModelName() << "\" input has " << input->Dims().size()
      << std::endl;
    exit(1);
  }

  // Input must be NHWC or NCHW...
  if ((*format != ni::ModelInput::FORMAT_NCHW) &&
      (*format != ni::ModelInput::FORMAT_NHWC)) {
    std::cerr
        << "unexpected input format " << ni::ModelInput_Format_Name(*format)
        << ", expecting "
        << ni::ModelInput_Format_Name(ni::ModelInput::FORMAT_NHWC) << " or "
        << ni::ModelInput_Format_Name(ni::ModelInput::FORMAT_NCHW)
        << std::endl;
    exit(1);
  }

  if (*format == ni::ModelInput::FORMAT_NHWC) {
    *h = input->Dims()[0];
    *w = input->Dims()[1];
    *c = input->Dims()[2];
  } else if (*format == ni::ModelInput::FORMAT_NCHW) {
    *c = input->Dims()[0];
    *h = input->Dims()[1];
    *w = input->Dims()[2];
  }
}

  void ResampleImage(const std::string& filename, size_t c, size_t h, size_t w,
      ni::ModelInput::Format format, const std::string& out_filename)
  {
    cv::Mat src_img = cv::imread(filename);
    if (src_img.empty()) {
      std::cerr << "error: unable to decode image " << filename << std::endl;
      exit(1);
    }

    cv::Size img_size(w, h);
    cv::Mat sample_resized;
    if (src_img.size() != img_size) {
      cv::resize(src_img, sample_resized, img_size);
    } else {
      sample_resized = src_img;
    }

    cv::imwrite(out_filename, sample_resized);
  }


} //namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  size_t batch_size = 1;
  size_t topk = 1;
  std::string preprocess_output_filename;
  std::string model_name;
  int model_version = -1;
  std::string url("localhost:8000");
  ProtocolType protocol = ProtocolType::HTTP;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:m:x:b:c:s:p:i:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'x':
        model_version = atoi(optarg);
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      case 'c':
        topk = atoi(optarg);
        break;
      case 'p':
        preprocess_output_filename = optarg;
        break;
      case 'i':
        protocol = ParseProtocol(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_name.empty()) { Usage(argv, "-m flag must be specified"); }
  if (batch_size <= 0) { Usage(argv, "batch size must be > 0"); }
  if (topk <= 0) { Usage(argv, "topk must be > 0"); }
  if (optind >= argc) {
    Usage(argv, "image file or image folder must be specified");
  }

  // Create the context for inference of the specified model. From it
  // extract and validate that the model meets the requirements for
  // image classification.
  std::unique_ptr<nic::InferContext> ctx;
  nic::Error err;
  if (protocol == ProtocolType::HTTP) {
    err = nic::InferHttpContext::Create(
      &ctx, url, model_name, model_version, verbose);
  } else {
    err = nic::InferGrpcContext::Create(
      &ctx, url, model_name, model_version, verbose);
  }
  if (!err.IsOk()) {
    std::cerr
      << "error: unable to create inference context: " << err << std::endl;
    exit(1);
  }

  size_t c, h, w;
  ni::ModelInput::Format format;
  ParseModel(ctx, batch_size, &c, &h, &w, &format, verbose);

  // Read the file(s) and preprocess them into input data accordingly
  std::vector<std::vector<std::string>> batched_filenames;
  struct stat name_stat;
  if (stat(argv[optind], &name_stat) == 0) {
    ResampleImage(std::string(argv[optind]), c, h, w, format, "/tmp/test.jpg");
  } else {
    std::cerr << "Failed to find '" << std::string(argv[optind]) << "': "
      << strerror(errno) << std::endl;
  }
  return 0;
}
