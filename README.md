# NVIDIA TensorRT Inference Server Clients

The NVIDIA TensorRT Inference Server provides a cloud inferencing
solution optimized for NVIDIA GPUs. The inference server provides
deep-learning inferencing via HTTP and gRPC endpoints, allowing remote
clients to request inferencing for any model being managed by the
server.

This repo contains C++ and Python client libraries that make it easy
to communicate with the inference server. Also included are a couple
of example applications.

- C++ and Python versions of *image\_client*, an example application
  that uses the C++ or Python client library to execute image
  classification models on the inference server.

- Python version of *grpc\_image\_client*, an example application that
  is functionally equivalent to *image\_client* but that uses gRPC
  generated client code to communicate with the inference server
  (instead of the inference server client library).

- C++ version of *perf\_client*, an example application that issues a
  large number of concurrent requests to the inference server to
  measure latency and throughput for a given model. You can use this
  to experiment with different model configuration settings for your
  models.

The inference server itself is delivered as a containerized solution
on the [NVIDIA GPU
Cloud](https://www.nvidia.com/en-us/gpu-cloud/). See the [Inference
Server User
Guide](https://docs.nvidia.com/deeplearning/sdk/inference-user-guide/index.html)
for information on how to install and configure the inference server.

Use [Issues](https://github.com/NVIDIA/dl-inference-server/issues) to
report any issues or questions about the client libraries and
examples. A [DevTalk
forum](https://devtalk.nvidia.com/default/board/262/container-inference-server)
is also available for inference server issues and questions.

## Branches

**master**: Active development branch. Typically will contain
 development corresponding to the next upcoming TensorRT Inference
 Server container release.

**yy.mm**: Branch compatible with TensorRT Inference Server yy.mm, for
  example *18.09*.

## Building the Clients

Before building the client libraries and applications you must first
install some prerequisites. The following instructions assume Ubuntu
16.04; other distributions will likely require additional/different
dependencies and Makefile modifications. For example Mint requires
that "-lssl -lcrypto" be added to IMAGE\_LDFLAGS and PERF\_LDFLAGS in
Makefile.clients.

OpenCV is used by image\_client to preprocess images before
sending them to the inference server for inferencing. The python-pil
package is required by the Python image\_client example.

    sudo apt-get update
    sudo apt-get install build-essential libcurl3-dev libopencv-dev libopencv-core-dev python-pil \
                         software-properties-common autoconf automake libtool pkg-config

Creating the whl file for the Python client library requires setuptools.
And grpcio-tools is required for gRPC support in Python client library.

    pip install --no-cache-dir --upgrade setuptools grpcio-tools

With those prerequisites installed, the C++ and Python client libraries
and example image\_client and perf\_client applications can be built:

    make -f Makefile.clients all pip

Build artifacts are in build/.  The Python whl file is generated in
build/dist/dist/ and can be installed with a command like the following:

    pip install --no-cache-dir --upgrade build/dist/dist/tensorrtserver-0.6.0-cp27-cp27mu-linux_x86_64.whl

## Building the Clients with Docker

A Dockerfile is provided for building the client libraries and examples
inside a container.

Before building, edit the Dockerfile to set PYVER to either 2.7 or 3.5
to select the version of Python that you will be using. The following
will build the C++ client library, C++ examples and create a Python
wheel file for the Python client library.

    $ docker build -t inference_server_clients .

The easiest way to extract the built libraries and examples from the
docker image is to mount a host driver into the container and then
copy the images.

    $ docker run --rm -it -v /tmp:/tmp/host inference_server_clients
    # cp build/image_client /tmp/host/.
    # cp build/perf_client /tmp/host/.
    # cp build/dist/dist/tensorrtserver-*.whl /tmp/host/.

You can now access image\_client, perf\_client and the wheel file from /tmp on the
host system.

## Image Classification Example

The image classification example that uses the C++ client API is
available at src/clients/c++/image\_client.cc. After building as
described above the executable is available at build/image\_client.
The python version of the image classification client is available at
src/clients/python/image\_client.py.

To use image\_client (or image\_client.py) you must first have an
inference server that is serving one or more image classification
models. The image\_client example requires that the model have a
single image input and produce a single classification output.

An example model store containing a single Caffe2 ResNet50 model
is provided in the examples/models directory that you can use to
demonstrate image\_client. Before using the example model store you
must fetch any missing model definition files from their public model
zoos.

    $ cd examples
    $ ./fetch_models.sh

Following the instructions in the [Inference Server User
Guide](https://docs.nvidia.com/deeplearning/sdk/inference-user-guide/index.html),
launch the inference server container pointing to that model
store. For example:

    $ nvidia-docker run --rm -p8000:8000 -p8001:8001 -v/path/to/dl-inference-server/examples/models:/models nvcr.io/nvidia/tensorrtserver:18.09-py3 trtserver --model-store=/models

Make sure you choose the most recent version of
nvcr.io/nvidia/tensorrtserver. Port 8000 exposes the inference server
HTTP endpoint and port 8001 exposes the gRPC endpoint. Replace
/path/to/dl-inference-server/examples/models with the corresponding
path in your local clone of this repo. Once the server is running you
can use image\_client application to send inference requests to the
server. You can specify a single image or a directory holding images.

    $ image_client -m resnet50_netdef -s INCEPTION examples/data/mug.jpg

The Python version of the application accepts the same command-line
arguments.

    $ src/clients/python/image_client.py -m resnet50_netdef -s INCEPTION examples/data/mug.jpg

The image\_client and image\_client.py examples use the inference
server client library to talk to the inference server. By default the
examples instruct the client library to use HTTP protocol to talk to
the inference server, but you can use gRPC protocol by providing the
-i flag. You must also use the -u flag to point at the gRPC endpoint
on the inference server.

    $ image_client -i grpc -u localhost:8001 -m resnet50_netdef -s INCEPTION examples/data/mug.jpg

By default the client prints the most probable classification for the image.

    Prediction totals:
            cnt=1   (504) COFFEE MUG

Use the -c flag to see more classifications.

    $ image_client -m resnet50_netdef -s INCEPTION -c 3 examples/data/mug.jpg
    Output probabilities:
    batch 0: 504 (COFFEE MUG) = 0.777365267277
    batch 0: 968 (CUP) = 0.213909029961
    batch 0: 967 (ESPRESSO) = 0.00294389552437
    Prediction totals:
            cnt=1	(504) COFFEE MUG

The -b flag allows you to send a batch of images for inferencing.  The
image\_client will form the batch from the image or images that you
specified. If the batch is bigger than the number of images then
image\_client will just repeat an image to fill the batch. This
example specifies a single image so you will see the same
classification results repeated (as indicated by the 'cnt' value).

    $ image_client -m resnet50_netdef -s INCEPTION -b 2 examples/data/mug.jpg
    Prediction totals:
            cnt=2	(504) COFFEE MUG

The grpc\_image\_client.py example behaves the same as the image\_client
examples except that instead of using the inference server client
library it uses the gRPC generated client library to communicate with
the inference server.

    $ src/clients/python/grpc_image_client.py -m resnet50_netdef -s INCEPTION -u localhost:8001 examples/data/mug.jpg


## Perf Example

The perf\_client example uses the C++ client API to send concurrent
requests to the inference server to measure latency and inferences per
second under varying client loads.  After building as described above,
the executable is available at build/perf\_client.

You can use perf\_client with any kind of model. It sends random data
as input and ignores the output. Any model in the example model store
can be used to demonstrate perf\_client. Before using the example
model store you must fetch any missing model definition files from
their public model zoos.

    $ cd examples
    $ ./fetch_models.sh

Following the instructions in the [Inference Server User
Guide](https://docs.nvidia.com/deeplearning/sdk/inference-user-guide/index.html),
launch the inference server container pointing to that model
store. For example:

    $ nvidia-docker run --rm -p8000:8000 -p8001:8001 -v/path/to/dl-inference-server/examples/models:/models nvcr.io/nvidia/tensorrtserver:18.09-py3 trtserver --model-store=/models

Make sure you choose the most recent version of
nvcr.io/nvidia/tensorrtserver. Port 8000 exposes the inference server
HTTP endpoint and port 8001 exposes the gRPC endpoint. Replace
/path/to/dl-inference-server/examples/models with the corresponding
path in your local clone of this repo.

The perf\_client example has two major modes. In the first mode you
specify how many concurrent clients you want to simulate and
perf\_client finds a stable latency and inferences/second for that
level of concurrency. Use the -t flag to control concurrency and -v to
see verbose output. For example, to simulate four clients continuously
sending requests to the inference server use the following
command-line.

    $ perf_client -m resnet50_netdef -p3000 -t4 -v
    *** Measurement Settings ***
      Batch size: 1
      Measurement window: 3000 msec

    Request concurrency: 4
      Pass [1] throughput: 238 infer/sec. Avg latency: 16767 usec (std 7150 usec)
      Pass [2] throughput: 238 infer/sec. Avg latency: 16781 usec (std 7176 usec)
      Pass [3] throughput: 238 infer/sec. Avg latency: 16751 usec (std 7192 usec)
      Client:
        Request count: 716
        Throughput: 238 infer/sec
        Avg latency: 16751 usec (standard deviation 7192 usec)
        Avg HTTP time: 16731 usec (send 723 usec + response wait 15982 usec + receive 26 usec)
      Server:
        Request count: 862
        Avg request latency: 14377 usec (ovOpenImageIOerhead 273 usec + wait 6502 usec + compute 7602 usec)

In the second mode perf\_client will generate a inferences/second
vs. latency curve by increasing concurrency until a specificy latency
limit is reached. This mode is enabled by using the -d option and -l
to specify the latency limit (you can also use -c to specify a maximum
concurrency limit).

    $ perf_client -m resnet50_netdef -p3000 -d -l15
    *** Measurement Settings ***
      Batch size: 1
      Measurement window: 3000 msec
      Latency limit: 15 msec

    Request concurrency: 1
      Client:
        Request count: 278
        Throughput: 92 infer/sec
        Avg latency: 10773 usec (standard deviation 609 usec)
        Avg HTTP time: 10736 usec (send/recv 778 usec + response wait 9958 usec)
      Server:
        Request count: 335
        Avg request latency: 7976 usec (overhead 187 usec + wait 0 usec + compute 7789 usec)

    Request concurrency: 2
      Client:
        Request count: 573
        Throughput: 191 infer/sec
        Avg latency: 10471 usec (standard deviation 837 usec)
        Avg HTTP time: 10436 usec (send/recv 852 usec + response wait 9584 usec)
      Server:
        Request count: 690
        Avg request latency: 7403 usec (overhead 238 usec + wait 11 usec + compute 7154 usec)

    Request concurrency: 3
      Client:
        Request count: 771
        Throughput: 257 infer/sec
        Avg latency: 11659 usec (standard deviation 2435 usec)
        Avg HTTP time: 11625 usec (send/recv 846 usec + response wait 10779 usec)
      Server:
        Request count: 928
        Avg request latency: 9466 usec (overhead 235 usec + wait 1946 usec + compute 7285 usec)

    Request concurrency: 4
      Client:
        Request count: 715
        Throughput: 238 infer/sec
        Avg latency: 16753 usec (standard deviation 7142 usec)
        Avg HTTP time: 16765 usec (send/recv 791 usec + response wait 15974 usec)
      Server:
        Request count: 858
        Avg request latency: 14370 usec (overhead 273 usec + wait 6467 usec + compute 7630 usec)

    [ 0] SUCCESS
    Inferences/Second vs. Client Average Batch Latency
    Concurrency: 1, 92 infer/sec, latency 10773 usec
    Concurrency: 2, 191 infer/sec, latency 10471 usec
    Concurrency: 3, 257 infer/sec, latency 11659 usec
    Concurrency: 4, 238 infer/sec, latency 16753 usec

Use the -f flag to generate a file containing CSV output of the
results.

    $ perf_client -m resnet50_netdef -p3000 -d -l15 -f perf.csv

You can then import the CSV file into a spreadsheet to help visualize
the latency vs inferences/second tradeoff as well as see some
components of the latency. Follow these steps:

- Open [this spreadsheet](https://docs.google.com/spreadsheets/d/1zszgmbSNHHXy0DVEU_4lrL4Md-6dUKwy_mLVmcseUrE)
- Make a copy from the File menu "Make a copy..."
- Open the copy
- Select the A2 cell
- From the File menu select "Import..."
- Select "Upload" and upload the file
- Select "Replace data at selected cell" and then select the "Import data" button


## C++ API

The C++ client API exposes a class-based interface for querying server
and model status and for performing inference. The commented interface
is available at src/clients/c++/request.h.

The following shows an example of the basic steps required for
inferencing (error checking not included to improve clarity, see
image_client.cc for full error checking):

```c++
// Create the context object for inferencing using the latest version
// of the 'mnist' model. Use the HTTP protocol for communication with
// the inference server (to use gRPC use InferGrpcContext::Create).
std::unique_ptr<InferContext> ctx;
InferHttpContext::Create(&ctx, "localhost:8000", "mnist");

// Get handle to model input and output.
std::shared_ptr<InferContext::Input> input;
ctx->GetInput(input_name, &input);

std::shared_ptr<InferContext::Output> output;
ctx->GetOutput(output_name, &output);

// Set options so that subsequent inference runs are for a given batch_size
// and return a result for ‘output’. The ‘output’ result is returned as a
// classification result of the ‘k’ most probable classes.
std::unique_ptr<InferContext::Options> options;
InferContext::Options::Create(&options);
options->SetBatchSize(batch_size);
options->AddClassResult(output, k);
ctx->SetRunOptions(*options);

// Provide input data for each batch.
input->Reset();
for (size_t i = 0; i < batch_size; ++i) {
  input->SetRaw(input_data[i]);
}

// Run inference and get the results. When the Run() call returns the ctx
// can be used for another inference run. Results are owned by the caller
// and can be retained as long as necessary.
std::vector<std::unique_ptr<InferContext::Result>> results;
ctx->Run(&results);

// For each entry in the batch print the top prediction.
for (size_t i = 0; i < batch_size; ++i) {
  InferContext::Result::ClassResult cls;
  results[0]->GetClassAtCursor(i, &cls);
  std::cout << "batch " << i << ": " << cls.label << std::endl;
}
```

## Python API

The Python client API provides similar capabilities as the C++
API. The commented interface for StatusContext and InferContext
classes is available at src/clients/python/\_\_init\_\_.py.

The following shows an example of the basic steps required for
inferencing (error checking not included to improve clarity):

```python
from tensorrtserver.api import *

# Create input with random data
input_list = list()
for b in range(batch_size):
    in = np.random.randint(size=input_size, dtype=input_dtype)
    input_list.append(in)

# Run inferencing and get the top-3 classes
ctx = InferContext("localhost:8000", ProtocolType.HTTP, "mnist")
results = ctx.run(
    { "data" : input_list },
    { "prob" : (InferContext.ResultFormat.CLASS, 3) },
    batch_size)


# Print results
for (result_name, result_val) in iteritems(results):
    for b in range(batch_size):
        print("output {}, batch {}: {}".format(result_name, b, result_val[b]))
```
