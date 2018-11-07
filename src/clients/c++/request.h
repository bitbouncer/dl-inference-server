// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <condition_variable>
#include <grpc++/grpc++.h>
//#include <grpcpp/grpcpp.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <curl/curl.h>
#include "src/core/api.pb.h"
#include "src/core/grpc_service.grpc.pb.h"
#include "src/core/grpc_service.pb.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config.h"
#include "src/core/request_status.pb.h"
#include "src/core/server_status.pb.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================
// Error
//
// Used by client API to report success or failure.
//
class Error {
public:
  // Create an error from a RequestStatus object.
  // @param status - the RequestStatus object
  explicit Error(const RequestStatus& status);

  // Create an error from a code.
  // @param code - The status code for the error
  explicit Error(RequestStatusCode code = RequestStatusCode::SUCCESS);

  // Create an error from a code and detailed message.
  // @param code - The status code for the error
  // @param msg - The detailed message for the error
  explicit Error(RequestStatusCode code, const std::string& msg);

  // @return the status code for the error.
  RequestStatusCode Code() const { return code_; }

  // @return the detailed messsage for the error. Empty if no detailed
  // message.
  const std::string& Message() const { return msg_; }

  // @return the ID of the inference server associated with this
  // error, or empty-string if no inference server is associated with
  // the error.
  const std::string& ServerId() const { return server_id_; }

  // @return the ID of the request associated with this error, or 0
  // (zero) if no request ID is associated with the error.
  uint64_t RequestId() const { return request_id_; }

  // @return true if this error is "ok"/"success", false if error
  // indicates a failure.
  bool IsOk() const { return code_ == RequestStatusCode::SUCCESS; }

  // Convenience "success" value can be used as Error::Success to
  // indicate no error.
  static const Error Success;

private:
  friend std::ostream& operator<<(std::ostream&, const Error&);
  RequestStatusCode code_;
  std::string msg_;
  std::string server_id_;
  uint64_t request_id_;
};

//==============================================================================
// ServerHealthContext
//
// A ServerHealthContext object is used to query an inference server
// for health information.s Once created a ServerHealthContext object
// can be used repeatedly to get health from the server.  A
// ServerHealthContext object can use either HTTP protocol or gRPC
// protocol depending on the Create function
// (ServerHealthHttpContext::Create or
// ServerHealthGrpcContext::Create). For example:
//
//   std::unique_ptr<ServerHealthContext> ctx;
//   ServerHealthHttpContext::Create(&ctx, "localhost:8000");
//   bool ready;
//   ctx->GetReady(&ready);
//   ...
//   bool live;
//   ctx->GetLive(&live);
//   ...
//
// Thread-safety:
//   ServerHealthContext::Create methods are thread-safe.
//   GetReady() and GetLive() are not thread-safe. For a given
//   ServerHealthContext, calls to GetReady() and GetLive() must be
//   serialized.
//
class ServerHealthContext {
public:
  // Contact the inference server and get readiness
  // @param ready - returns the readiness
  // @return Error object indicating success or failure
  virtual Error GetReady(bool* ready) = 0;

  // Contact the inference server and get liveness
  // @param live - returns the liveness
  // @return Error object indicating success or failure
  virtual Error GetLive(bool* live) = 0;

protected:
  ServerHealthContext(bool);

  // If true print verbose output
  const bool verbose_;
};

//==============================================================================
// ServerStatusContext
//
// A ServerStatusContext object is used to query an inference server
// for status information, including information about the models
// available on that server. Once created a ServerStatusContext object
// can be used repeatedly to get status from the server.
// A ServerStatusContext object can use either HTTP protocol or gRPC protocol
// depending on the Create function (ServerStatusHttpContext::Create or
// ServerStatusGrpcContext::Create). For example:
//
//   std::unique_ptr<ServerStatusContext> ctx;
//   ServerStatusHttpContext::Create(&ctx, "localhost:8000");
//   ServerStatus status;
//   ctx->GetServerStatus(&status);
//   ...
//   ctx->GetServerStatus(&status);
//   ...
//
// Thread-safety:
//   ServerStatusContext::Create methods are thread-safe.
//   GetServerStatus() is not thread-safe. For a given
//   ServerStatusContext, calls to GetServerStatus() must be
//   serialized.
//
class ServerStatusContext {
public:
  // Contact the inference server and get status
  // @param status - returns the status
  // @return Error object indicating success or failure
  virtual Error GetServerStatus(ServerStatus* status) = 0;

protected:
  ServerStatusContext(bool);

  // If true print verbose output
  const bool verbose_;
};

//==============================================================================
// InferContext
//
// An InferContext object is used to run inference on an inference
// server for a specific model. Once created an InferContext object
// can be used repeatedly to perform inference using the
// model. Options that control how inference is performed can be
// changed in between inference runs.
// A InferContext object can use either HTTP protocol or gRPC protocol
// depending on the Create function (InferHttpContext::Create or
// InferGrpcContext::Create). For example:
//
//   std::unique_ptr<InferContext> ctx;
//   InferHttpContext::Create(&ctx, "localhost:8000", "mnist");
//   ...
//   std::unique_ptr<Options> options0;
//   Options::Create(&options0);
//   options->SetBatchSize(b);
//   options->AddClassResult(output, topk);
//   ctx->SetRunOptions(*options0);
//   ...
//   ctx->Run(&results0);  // run using options0
//   ctx->Run(&results1);  // run using options0
//   ...
//   std::unique_ptr<Options> options1;
//   Options::Create(&options1);
//   options->AddRawResult(output);
//   ctx->SetRunOptions(*options);
//   ...
//   ctx->Run(&results2);  // run using options1
//   ctx->Run(&results3);  // run using options1
//   ...
//
// Note that the Run() calls are not thread-safe but a new Run() can
// be invoked as soon as the previous completes. The returned result
// objects are owned by the caller and may be retained and accessed
// even after the InferContext object is destroyed.
//
// Also note that AsyncRun() and GetAsyncRunStatus() calls are not thread-safe.
// What's more, calling one method while the other one is running will result in
// undefined behavior given that they will modify the shared data internally.
//
// For more parallelism multiple InferContext objects can access the
// same inference server with no serialization requirements across
// those objects.
//
// Thread-safety:
//   InferContext::Create methods are thread-safe.
//   All other InferContext methods, and nested class methods are not
//   thread-safe.
//
class InferContext {
public:
  //==============
  // Input
  // An input to the model being used for inference.
  class Input {
  public:
    virtual ~Input() { };

    // @return the name of the input.
    virtual const std::string& Name() const = 0;

    // @return the size in bytes for a single instance of this input
    // (that is, the size when batch-size == 1).
    virtual size_t ByteSize() const = 0;

    // @return the data type of the input.
    virtual DataType DType() const = 0;

    // @return the format of the input.
    virtual ModelInput::Format Format() const = 0;

    // @return the dimensions of the input.
    virtual const DimsList& Dims() const = 0;

    // Prepare this input to receive new tensor values. Forget any
    // existing values that were set by previous calls to
    // Input::SetRaw().
    // @return Error object indicating success or failure
    virtual Error Reset() = 0;

    // Set tensor values for this input from a byte array. The array
    // is not copied and so the it must not be modified or destroyed
    // until this input is no longer needed (that is until the Run()
    // call(s) that use the input have completed). For batched inputs
    // this function must be called batch-size times to provide all
    // tensor values for a batch of this input.
    // @param input - pointer to the array holding tensor value
    // @param input_byte_size - size of the array in bytes, must match
    // the size expected by the input.
    // @return Error object indicating success or failure
    virtual Error SetRaw(
      const uint8_t* input, size_t input_byte_size) = 0;

    // Set tensor values for this input from a byte vector. The vector
    // is not copied and so the it must not be modified or destroyed
    // until this input is no longer needed (that is until the Run()
    // call(s) that use the input have completed). For batched inputs
    // this function must be called batch-size times to provide all
    // tensor values for a batch of this input.
    // @param input - vector holding tensor values
    // @return Error object indicating success or failure
    virtual Error SetRaw(const std::vector<uint8_t>& input) = 0;
  };

  //==============
  // Output
  // An output from the model being used for inference.
  class Output {
  public:
    virtual ~Output() { };

    // @return the name of the output.
    virtual const std::string& Name() const = 0;

    // @return the size in bytes for a single instance of this output
    // (that is, the size when batch-size == 1).
    virtual size_t ByteSize() const = 0;

    // @return the data type of the output.
    virtual DataType DType() const = 0;

    // @return the dimensions of the output.
    virtual const DimsList& Dims() const = 0;
  };

  //==============
  // Result
  // An inference result corresponding to an output.
  class Result {
  public:
    virtual ~Result() { };

    // Format in which result is returned. RAW format is the entire
    // result tensor of values. CLASS format is the top-k highest
    // probability values of the result and the associated class label
    // (if provided by the model).
    enum ResultFormat { RAW = 0, CLASS = 1 };

    // @return the name of the model that produced this result.
    virtual const std::string& ModelName() const = 0;

    // @return the version of the model that produced this result.
    virtual uint32_t ModelVersion() const = 0;

    // @return the Output object corresponding to this result.
    virtual const std::shared_ptr<Output> GetOutput() const = 0;

    // Get a reference to entire raw result data for a specific batch
    // entry. Returns error if this result is not RAW format.
    // @param batch_idx - return results for this entry the batch
    // @param buf - returns the vector of result bytes
    // @return Error object indicating success or failure
    virtual Error
      GetRaw(size_t batch_idx, const std::vector<uint8_t>** buf) const = 0;

    // Get a reference to raw result data for a specific batch entry
    // at the current "cursor" and advance the cursor by the specified
    // number of bytes. More typically use GetRawAtCursor<T>() method
    // to return the data as a specific type T. Use ResetCursor() to
    // reset the cursor to the beginning of the result. Returns error
    // if this result is not RAW format.
    // @param batch_idx - return results for this entry the batch
    // @param buf - returns pointer to 'adv_byte_size' bytes of data
    // @param adv_byte_size - number of bytes of data to get reference to
    // @return Error object indicating success or failure
    virtual Error GetRawAtCursor(
      size_t batch_idx, const uint8_t** buf, size_t adv_byte_size) = 0;

    // Read a value for a specific batch entry at the current "cursor"
    // from the result tensor as the specified type T and advance the
    // cursor. Use ResetCursor() to reset the cursor to the beginning
    // of the result. Returns error if this result is not RAW format.
    // @param batch_idx - return results for this entry the batch
    // @param out - returns the value
    // @return Error object indicating success or failure
    template<typename T>
    Error GetRawAtCursor(size_t batch_idx, T* out);

    // ClassResult
    // The result value for CLASS format results indicating the index
    // of the class in the result vector, the value of the class (as a
    // float) and the corresponding label (if provided by the model).
    struct ClassResult {
      size_t idx;
      float value;
      std::string label;
    };

    // Get the number of class results for a batch. Returns error if
    // this result is not CLASS format.
    // @param batch_idx - return results for this entry the batch
    // @param cnt - returns the number of ClassResult entries
    // @return Error object indicating success or failure
    virtual Error GetClassCount(size_t batch_idx, size_t* cnt) const = 0;

    // Get the ClassResult result for a specific batch entry at the
    // current cursor. Use ResetCursor() to reset the cursor to the
    // beginning of the result. Returns error if this result is not
    // CLASS format.
    // @param batch_idx - return results for this entry the batch
    // @param result - returns the ClassResult value
    // @return Error object indicating success or failure
    virtual Error GetClassAtCursor(size_t batch_idx, ClassResult* result) = 0;

    // Reset cursor to beginning of result for all batch entries.
    // @return Error object indicating success or failure
    virtual Error ResetCursors() = 0;

    // Reset cursor to beginning of result for specified batch entry.
    // @param batch_idx - result cursor for this entry the batch
    // @return Error object indicating success or failure
    virtual Error ResetCursor(size_t batch_idx) = 0;
  };

  //==============
  // Options
  // Run options to be applied to all subsequent Run() invocations.
  class Options {
  public:
    virtual ~Options() { };

    // Create a new Options object with default values.
    // @return Error object indicating success or failure.
    static Error Create(std::unique_ptr<Options>* options);

    // @return the batch size to use for all subsequent inferences.
    virtual size_t BatchSize() const = 0;

    // Set the batch size to use for all subsequent inferences.
    // @param batch_size - the batch size
    virtual void SetBatchSize(size_t batch_size) = 0;

    // Add 'output' to the list of requested RAW results. Run() will
    // return the output's full tensor as a result.
    // @param output - the output
    // @return Error object indicating success or failure
    virtual Error AddRawResult(
      const std::shared_ptr<InferContext::Output>& output) = 0;

    // Add 'output' to the list of requested CLASS results. Run() will
    // return the highest 'k' values of 'output' as a result.
    // @param output - the output
    // @param k - return 'k' class results for the output
    // @return Error object indicating success or failure
    virtual Error AddClassResult(
      const std::shared_ptr<InferContext::Output>& output, uint64_t k) = 0;
  };

  //==============
  // Request
  // Handle to a inference request, which will be used to get request results
  // if the request is sent by asynchronous functions.
  //
  // Depend on the protocol, Request class will have different implementations.
  // But all implementations should contain data required until the request is
  // completed and the placeholder for requested results.
  // The InferContext is responsible for initializing the Request, sending it,
  // monitoring the request transfer and retrieving results from corresponding
  // request on demand.
  class Request {
  public:
    virtual ~Request() = default;

    // @return the unique identifier of the request.
    virtual uint64_t Id() const = 0;
  };

  //==============
  // Stat
  // Struct contains cumulative statistic of the InferContext
  // Note: for gRPC protocol,
  //   'cumulative_send_time_ns' represent the time for
  //   marshalling infer request.
  //   'cumulative_receive_time_ns' represent the time for
  //   unmarshalling infer response.
  struct Stat {
    // Total number of request completed
    size_t completed_request_count;
    // Time from the request start until the response is completely received
    uint64_t cumulative_total_request_time_ns;
    // Time from the request start until the last byte is sent
    uint64_t cumulative_send_time_ns;
    // Time from receiving first byte of the response until the response
    // is completely received
    uint64_t cumulative_receive_time_ns;

    Stat()
     : completed_request_count(0), cumulative_total_request_time_ns(0),
       cumulative_send_time_ns(0), cumulative_receive_time_ns(0) {}
  };

  //==============
  // RequestTimers
  // Timer to record the time for different request stages
  class RequestTimers {
  public:
    enum Kind {
      REQUEST_START,
      REQUEST_END,
      SEND_START,
      SEND_END,
      RECEIVE_START,
      RECEIVE_END
    };

    RequestTimers();

    // Reset the timer values to 0. Should always be called before re-using
    // the timer
    Error Reset();

    // Record the current timestamp for the request stage specified by 'kind'
    Error Record(Kind kind);
  private:
    friend class InferContext;
    friend class InferHttpContext;
    friend class InferGrpcContext;
    struct timespec request_start_;
    struct timespec request_end_;
    struct timespec send_start_;
    struct timespec send_end_;
    struct timespec receive_start_;
    struct timespec receive_end_;
  };

public:
  virtual ~InferContext() = default;

  // @return the model name being used for inference.
  const std::string& ModelName() const { return model_name_; }

  // @return the model version being used for inference. -1 indicates
  // that the latest (i.e. highest version number) version of that
  // model is being used.
  int ModelVersion() const { return model_version_; }

  // @return the maximum batch size supported by the context.
  uint64_t MaxBatchSize() const { return max_batch_size_; }

  // @return the inputs of the model.
  const std::vector<std::shared_ptr<Input>>& Inputs() const {
    return inputs_;
  }

  // @return the outputs of the model.
  const std::vector<std::shared_ptr<Output>>& Outputs() const {
    return outputs_;
  }

  // Get a named input.
  // @param name - the name of the input
  // @param input - returns the Input object for 'name'
  // @return Error object indicating success or failure
  Error GetInput(
    const std::string& name, std::shared_ptr<Input>* input) const;

  // Get a named output.
  // @param name - the name of the output
  // @param input - returns the Output object for 'name'
  // @return Error object indicating success or failure
  Error GetOutput(
    const std::string& name, std::shared_ptr<Output>* output) const;

  // Set the options to use for all subsequent Run() invocations.
  // @param options - the options
  // @return Error object indicating success or failure
  Error SetRunOptions(const Options& options);

  // Get the current statistic of the InferContext. 
  // @parm stat - returns Stat objects holding InferContext statistic.
  // @return Error object indicating success or failure
  Error GetStat(Stat* stat);

  // Send a request to the inference server to perform an inference to
  // produce a result for the outputs specified in the most recent
  // call to SetRunOptions(). The Result objects holding the output
  // values are returned in the same order as the outputs are
  // specified in the options.
  // @param results - returns Result objects holding inference results.
  // @return Error object indicating success or failure
  virtual Error Run(std::vector<std::unique_ptr<Result>>* results) = 0;

  // Send an asynchronous request to the inference server to perform
  // an inference to produce a result for the outputs specified in the most
  // recent call to SetRunOptions().
  // @param async_request - returns Request objects as handle to retrieve
  // inference results.
  // @return Error object indicating success or failure
  virtual Error AsyncRun(
    std::shared_ptr<Request>* async_request) = 0;

  // Get the result of the asynchronous request referenced by 'async_request'.
  // The Result objects holding the output values are returned in the same order
  // as the outputs are specified in the options when AsyncRun() was called.
  // @param results - returns Result objects holding inference results.
  // @param async_request - Request objects as handle to retrieve results.
  // @param wait - if true, block until the request completes. Otherwise, return
  // immediately.
  // @return Error object indicating success or failure. Success will be
  // returned only if the request has been completed succesfully. UNAVAILABLE
  // will be returned if 'wait' is false and the request is not ready.
  virtual Error GetAsyncRunResults(
    std::vector<std::unique_ptr<Result>>* results,
    const std::shared_ptr<Request>& async_request, bool wait) = 0;

  // Get any one completed asynchronous request.
  // The Request object will contain the request that is completed
  // and waiting to give the result.
  // @param request - returns Request objects holding completed request.
  // @param wait - if true, block until one request completes. Otherwise, return
  // immediately.
  // @return Error object indicating success or failure. Success will be
  // returned only if the request has been completed succesfully. UNAVAILABLE
  // will be returned if 'wait' is false and no request is ready.
  Error GetReadyAsyncRequest(
    std::shared_ptr<Request>* async_request, bool wait);

protected:
  InferContext(const std::string&, int, bool);

  // Function for worker thread to proceed the data transfer for all requests
  virtual void AsyncTransfer() = 0;

  // Helper function called before inference to prepare 'request'
  virtual Error PreRunProcessing(std::shared_ptr<Request>& request) = 0;

  // Helper function called by GetAsyncRunResults() to check if the request
  // is ready. If the request is valid and wait == true,
  // the function will block until request is ready.
  Error IsRequestReady(
    const std::shared_ptr<Request>& async_request, bool wait);

  // Update the context stat with the given timer
  Error UpdateStat(const RequestTimers& timer);

  using AsyncReqMap = std::map<uintptr_t, std::shared_ptr<Request>>;

  // map to record ongoing asynchronous requests with pointer to easy handle
  // as key
  AsyncReqMap ongoing_async_requests_;

  // Model name
  const std::string model_name_;

  // Model version
  const int model_version_;

  // If true print verbose output
  const bool verbose_;

  // Maximum batch size supported by this context.
  uint64_t max_batch_size_;

  // Total size of all inputs, in bytes (must be 64-bit integer
  // because used with curl_easy_setopt).
  uint64_t total_input_byte_size_;

  // Requested batch size for inference request
  uint64_t batch_size_;

  // Use to assign unique identifier for each asynchronous request
  uint64_t async_request_id_;

  // The inputs and outputs
  std::vector<std::shared_ptr<Input>> inputs_;
  std::vector<std::shared_ptr<Output>> outputs_;

  // Settings generated by current option
  // InferRequestHeader protobuf describing the request
  InferRequestHeader infer_request_;

  // Outputs requested for inference request
  std::vector<std::shared_ptr<Output>> requested_outputs_;

  //Standalone request context used for synchronous request
  std::shared_ptr<Request> sync_request_;

  // The statistic of the current context
  Stat context_stat_;

  // worker thread that will perform the asynchronous transfer
  std::thread worker_;

  // Avoid race condition between main thread and worker thread
  std::mutex mutex_;

  // Condition variable used for waiting on asynchronous request
  std::condition_variable cv_;

  // signal for worker thread to stop
  bool exiting_;
};

//==============================================================================
// ProfileContext
//
// A ProfileContext object is used to control profiling on the
// inference server. Once created a ProfileContext object can be used
// repeatedly.
// A ProfileContext object can use either HTTP protocol or gRPC protocol
// depending on the Create function (ProfileHttpContext::Create or
// ProfileGrpcContext::Create). For example:
//
//   std::unique_ptr<ProfileContext> ctx;
//   ProfileGrpcContext::Create(&ctx, "localhost:8000");
//   ctx->StartProfile();
//   ...
//   ctx->StopProfile();
//   ...
//
// Thread-safety:
//   ProfileContext::Create methods are thread-safe.  StartProfiling()
//   and StopProfiling() are not thread-safe. For a given
//   ProfileContext, calls to these methods must be serialized.
//
class ProfileContext {
public:
  // Start profiling on the inference server
  // @return Error object indicating success or failure
  Error StartProfile();

  // Start profiling on the inference server
  // @return Error object indicating success or failure
  Error StopProfile();

protected:
  ProfileContext(bool);
  virtual Error SendCommand(const std::string& cmd_str) = 0;

  // If true print verbose output
  const bool verbose_;
};

//==============================================================================
// ServerHealthHttpContext
//
// A ServerHealthHttpContext object is the HTTP instantiation of the
// ServerHealthContext class, please refer ServerHealthContext class
// for detail usage.
//
class ServerHealthHttpContext : public ServerHealthContext {
public:
  // Create context that returns health information.
  // @param ctx - returns the new ServerHealthHttpContext object
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ServerHealthContext>* ctx,
    const std::string& server_url, bool verbose = false);

  // @see ServerHealthContext::GetReady
  Error GetReady(bool* ready) override;

  // @see ServerHealthContext::GetLive
  Error GetLive(bool* live) override;

private:
  ServerHealthHttpContext(const std::string&, bool);
  Error GetHealth(const std::string& url, bool* health);

  // URL for health endpoint on inference server.
  const std::string url_;
};

//==============================================================================
// ServerStatusHttpContext
//
// A ServerStatusHttpContext object is the HTTP instantiation of
// the ServerStatusContext class, please refer ServerStatusContext class for
// detail usage.
//
class ServerStatusHttpContext : public ServerStatusContext {
public:
  // Create context that returns information about server and all
  // models on the server using HTTP protocol.
  // @param ctx - returns the new ServerStatusHttpContext object
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ServerStatusContext>* ctx,
    const std::string& server_url, bool verbose = false);

  // Create context that returns information about server and one
  // model using HTTP protocol.
  // @param ctx - returns the new ServerStatusHttpContext object
  // @param server_url - inference server name and port
  // @param model-name - get information for this model
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ServerStatusContext>* ctx,
    const std::string& server_url, const std::string& model_name,
    bool verbose = false);

  // Contact the inference server and get status
  // @param status - returns the status
  // @return Error object indicating success or failure
  Error GetServerStatus(ServerStatus* status) override;

private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  static size_t ResponseHandler(void*, size_t, size_t, void*);

  ServerStatusHttpContext(const std::string&, bool);
  ServerStatusHttpContext(const std::string&, const std::string&, bool);

  // URL for status endpoint on inference server.
  const std::string url_;

  // RequestStatus received in server response
  RequestStatus request_status_;

  // Serialized ServerStatus response from server.
  std::string response_;
};

//==============================================================================
// InferHttpContext
//
// An InferHttpContext object is the HTTP instantiation of
// the InferContext class, please refer InferContext class for
// detail usage.
//
class InferHttpContext : public InferContext {
public:
  ~InferHttpContext() override;

  // Create context that performs inference for a model using HTTP protocol.
  // @param ctx - returns the new InferHttpContext object
  // @param server_url - inference server name and port
  // @param model_name - name of the model to use for inference
  // @param model_version - version of the model to use for inference,
  // or -1 to indicate that the latest (i.e. highest version number)
  // version should be used
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::string& model_name, int model_version = -1,
    bool verbose = false);

  // @see InferContext.Run()
  Error Run(std::vector<std::unique_ptr<Result>>* results) override;

  // @see InferContext.AsyncRun()
  Error AsyncRun(
    std::shared_ptr<Request>* async_request) override;

  // @see InferContext.GetAsyncRunResults()
  Error GetAsyncRunResults(
    std::vector<std::unique_ptr<Result>>* results,
    const std::shared_ptr<Request>& async_request, bool wait) override;

private:
  static size_t RequestProvider(void*, size_t, size_t, void*);
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  static size_t ResponseHandler(void*, size_t, size_t, void*);

  InferHttpContext(
    const std::string&, const std::string&, int, bool);

  // @see InferContext.AsyncTransfer()
  void AsyncTransfer() override;

  // @see InferContext.PreRunProcessing()
  Error PreRunProcessing(std::shared_ptr<Request>& request) override;

  // curl multi handle for processing asynchronous requests
  CURLM* multi_handle_;

  // URL to POST to
  std::string url_;

  // Serialized InferRequestHeader
  std::string infer_request_str_;

  // Keep an easy handle alive to reuse the connection
  CURL* curl_;
};

//==============================================================================
// ProfileHttpContext
//
// A ProfileHttpContext object is the HTTP instantiation of
// the ProfileContext class, please refer ProfileContext class for
// detail usage.
//
class ProfileHttpContext : public ProfileContext {
public:
  // Create context that controls profiling on a server using HTTP protocol.
  // @param ctx - returns the new ProfileContext object
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ProfileContext>* ctx, const std::string& server_url,
    bool verbose = false);

private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);

  ProfileHttpContext(const std::string&, bool);
  Error SendCommand(const std::string& cmd_str) override;

  // URL for status endpoint on inference server.
  const std::string url_;

  // RequestStatus received in server response
  RequestStatus request_status_;
};

//==============================================================================
// ServerHealthGrpcContext
//
// A ServerHealthGrpcContext object is the gRPC instantiation of
// the ServerHealthContext class, please refer ServerHealthContext class for
// detail usage.
//
class ServerHealthGrpcContext : public ServerHealthContext {
public:
  // Create context that returns health information about server.
  // @param ctx - returns the new ServerHealthGrpcContext object
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ServerHealthContext>* ctx,
    const std::string& server_url, bool verbose = false);

  // @see ServerHealthContext::GetReady
  Error GetReady(bool* ready) override;

  // @see ServerHealthContext::GetLive
  Error GetLive(bool* live) override;

private:
  ServerHealthGrpcContext(const std::string&, bool);
  Error GetHealth(const std::string& mode, bool* health);

  // gRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;
};

//==============================================================================
// ServerStatusGrpcContext
//
// A ServerStatusGrpcContext object is the gRPC instantiation of
// the ServerStatusContext class, please refer ServerStatusContext class for
// detail usage.
//
class ServerStatusGrpcContext : public ServerStatusContext {
public:
  // Create context that returns information about server and all
  // models on the server using gRPC protocol.
  // @param ctx - returns the new ServerStatusGrpcContext object
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ServerStatusContext>* ctx,
    const std::string& server_url, bool verbose = false);

  // Create context that returns information about server and one
  // model using gRPC protocol.
  // @param ctx - returns the new ServerStatusGrpcContext object
  // @param server_url - inference server name and port
  // @param model-name - get information for this model
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ServerStatusContext>* ctx,
    const std::string& server_url, const std::string& model_name,
    bool verbose = false);

  // Contact the inference server and get status
  // @param status - returns the status
  // @return Error object indicating success or failure
  Error GetServerStatus(ServerStatus* status) override;

private:
  ServerStatusGrpcContext(const std::string&, bool);
  ServerStatusGrpcContext(const std::string&, const std::string&, bool);

  // Model name
  const std::string model_name_;

  // gRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;
};

//==============================================================================
// InferGrpcContext
//
// A InferGrpcContext object is the gRPC instantiation of
// the InferContext class, please refer InferContext class for
// detail usage.
//
class InferGrpcContext : public InferContext {
public:
  ~InferGrpcContext() override;

  // Create context that performs inference for a model using gRPC protocol.
  // @param ctx - returns the new InferContext object
  // @param server_url - inference server name and port
  // @param model_name - name of the model to use for inference
  // @param model_version - version of the model to use for inference,
  // or -1 to indicate that the latest (i.e. highest version number)
  // version should be used
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::string& model_name, int model_version = -1,
    bool verbose = false);

  // @see InferContext.Run()
  Error Run(std::vector<std::unique_ptr<Result>>* results) override;

  // @see InferContext.AsyncRun()
  Error AsyncRun(
    std::shared_ptr<Request>* async_request) override;

  // @see InferContext.GetAsyncRunResults()
  Error GetAsyncRunResults(
    std::vector<std::unique_ptr<Result>>* results,
    const std::shared_ptr<Request>& async_request, bool wait) override;

private:
  InferGrpcContext(
    const std::string&, const std::string&, int, bool);

  // @see InferContext.AsyncTransfer()
  void AsyncTransfer() override;

  // @see InferContext.PreRunProcessing()
  Error PreRunProcessing(std::shared_ptr<Request>& request) override;

  // additional vector contains 1-indexed key to available slots
  // in async request map.
  std::vector<uintptr_t> reusable_slot_;

  // The producer-consumer queue used to communicate asynchronously with
  // the gRPC runtime.
  grpc::CompletionQueue async_request_completion_queue_;

  // gRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_; 

  // request for gRPC call, one request object can be used for multiple calls
  // since it can be overwritten as soon as the gRPC send finishes.
  InferRequest request_;
};

//==============================================================================
// ProfileGrpcContext
//
// A ProfileGrpcContext object is the gRPC instantiation of
// the ProfileContext class, please refer ProfileContext class for
// detail usage.
//
class ProfileGrpcContext : public ProfileContext {
public:
  // Create context that controls profiling on a server using gRPC protocol.
  // @param ctx - returns the new ProfileContext object
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  // @return Error object indicating success or failure.
  static Error Create(
    std::unique_ptr<ProfileContext>* ctx, const std::string& server_url,
    bool verbose = false);

private:
  ProfileGrpcContext(const std::string&, bool);
  Error SendCommand(const std::string& cmd_str) override;

  // gRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;
};

//==============================================================================

std::ostream& operator<<(std::ostream&, const Error&);

template<typename T>
Error
InferContext::Result::GetRawAtCursor(size_t batch_idx, T* out)
{
  const uint8_t* buf;
  Error err = GetRawAtCursor(batch_idx, &buf, sizeof(T));
  if (!err.IsOk()) {
    return err;
  }

  std::copy(buf, buf + sizeof(T), reinterpret_cast<uint8_t*>(out));
  return Error::Success;
}

}}} // namespace nvidia::inferenceserver::client
