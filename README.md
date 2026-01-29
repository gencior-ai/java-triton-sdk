[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Java Triton SDK

Java Triton SDK is a high-performance, lightweight library designed to bridge the gap between Java applications and the NVIDIA Triton Inference Server.

While Python has excellent support for Triton, Java enterprise environments often struggle with complex gRPC/Protobuf boilerplate. This SDK simplifies that process, allowing you to focus on your AI logic rather than networking protocols.

## Roadmap & Planned Features

### Phase 1: Complete gRPC Implementation

#### Trace & Logging Management
- [ ] **Update trace settings** - Modify trace settings for a specific model or globally
- [ ] **Get trace settings** - Retrieve trace settings for a model or global configuration
- [ ] **Update global log settings** - Configure global logging levels and targets
- [ ] **Get global log settings** - Retrieve current global logging configuration

#### Memory Management
- [ ] **System shared memory management** (global and CUDA)
  - Register/unregister system shared memory regions
  - Query available shared memory
  - Manage memory buffers across GPU and CPU
- [ ] **CUDA shared memory management**
  - CUDA device memory registration
  - GPU memory allocation and deallocation
  - Memory synchronization utilities

#### Advanced Streaming & Communication
- [ ] **Streaming Inference Input abstraction** - High-level API for streaming input data
- [ ] **Bidirectional streaming** - Full support for streaming inference requests/responses from client
- [ ] **Requested Output management** - Control which outputs are returned in inference responses
- [ ] **Secure communication (TLS)** - Support for TLS/SSL encrypted connections (currently not supported)

---

### Phase 2: HTTP/HTTPS Client Implementation

Provide full feature parity with gRPC implementation through HTTP/HTTPS protocol:

#### Core Inference
- [ ] HTTP/HTTPS synchronous inference requests
- [ ] HTTP/HTTPS asynchronous inference requests
- [ ] Custom parameter passing via HTTP

#### Server Management
- [ ] Server health checks (live/ready) over HTTP
- [ ] Server metadata retrieval
- [ ] Model repository index queries
- [ ] Model load/unload operations

#### Model Management
- [ ] Model metadata queries
- [ ] Model configuration retrieval
- [ ] Model statistics and monitoring

#### Advanced Features
- [ ] Trace and logging management over HTTP
- [ ] Memory management APIs
- [ ] Streaming inference support
- [ ] Requested output filtering
- [ ] TLS/SSL support for HTTPS

---

### Implementation Notes

- **API Consistency**: HTTP/HTTPS client will maintain identical method signatures as gRPC implementation for seamless switching
- **Configuration**: Single configuration object to support both gRPC and HTTP protocols
- **Performance**: HTTP client optimized for scenarios where gRPC is not available or not preferred
- **Security**: TLS support will be added to both gRPC and HTTP implementations
- **Backwards Compatibility**: New features will be added without breaking existing API