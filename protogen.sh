protoc src/core/api.proto -I=. --cpp_out=.
protoc src/core/grpc_service.proto -I=. --cpp_out=. 
protoc src/core/request_status.proto -I=. --cpp_out=.
protoc src/core/server_status.proto -I=. --cpp_out=.
protoc src/core/model_config.proto -I=. --cpp_out=.


protoc src/core/grpc_service.proto -I=. --grpc_out=. --plugin=protoc-gen-grpc=/usr/local/bin/grpc_cpp_plugin

