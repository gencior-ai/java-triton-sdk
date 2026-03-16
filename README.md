[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Java Triton SDK

Java Triton SDK is a high-performance, lightweight library designed to bridge the gap between Java applications and the NVIDIA Triton Inference Server.

While Python has excellent support for Triton, Java enterprise environments often struggle with complex gRPC/Protobuf boilerplate. This SDK simplifies that process, allowing you to focus on your AI logic rather than networking protocols.

## Testing

### Unit Tests

```bash
mvn test
```

### Integration Tests

Integration tests use [Testcontainers](https://www.testcontainers.org/) to spin up a real Triton Inference Server with Python backend models. **Docker is required**.

```bash
mvn verify
```

Five Python backend models are provided in `dev/models_cpu/` for integration testing:

| Model | Inputs | Output | Purpose |
|-------|--------|--------|---------|
| `identity_fp32` | FP32 | FP32 | Echo identity for FP32 inference |
| `identity_int32` | INT32 | INT32 | Echo identity for INT32 inference |
| `identity_string` | STRING | STRING | Echo identity for string inference |
| `adder` | 2x INT32 | INT32 | Sum of two inputs (multi-input) |
| `sleeper` | FP32 + DELAY_MS | FP32 | Configurable delay (async/timing) |
| `streaming_echo` | STRING | STRING | Decoupled model, streams words as tokens (LLM simulation) |

> **Note:** Integration tests are skipped in CI (`-DskipITs`) because the Triton Docker image is too large for GitHub Actions runners.

## Roadmap & Planned Features

### Phase 1: gRPC Implementation

#### Server Management
- [x] **Server health checks** - `isServerLive()`, `isServerReady()`
- [x] **Server metadata** - `getServerMetadata()` (name, version, extensions)

#### Model Management
- [x] **Model readiness** - `isModelReady()` with optional version
- [x] **Model metadata** - `getModelMetadata()` (inputs/outputs schema, versions)
- [x] **Model configuration** - `getModelConfig()` (platform, backend, batch size)
- [x] **Model repository index** - `getModelRepositoryIndex()` (list all models and states)
- [x] **Model load/unload** - `loadModel()`, `unLoadModel()`
- [x] **Inference statistics** - `getInferenceStatistics()` (counts, latency, queue, cache)

#### Core Inference
- [x] **Synchronous inference** - `infer()` with inputs, version, and custom parameters
- [x] **Asynchronous inference** - `inferAsync()` returning `CompletableFuture<InferResult>`
- [x] **Custom parameters** - `InferParameters` builder with type-safe values
- [x] **Full data type support** - INT8/16/32/64, UINT8/16/32/64, FP16/32/64, BF16, BOOL, BYTES
- [x] **Streaming inference** - `inferStream()` with callback listener and `inferStreamPublisher()` with `Flow.Publisher` for LLM token-by-token generation (decoupled models)

#### Logging
- [x] **Structured logging** - SLF4J with DEBUG/TRACE/ERROR levels and operation timing

#### Testing
- [x] **Unit tests** - Full coverage with Mockito mocks (171 tests)
- [x] **Integration tests** - Testcontainers with real Triton server and Python backend models

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
- [x] **Streaming Inference** - Callback-based (`InferStreamListener`) and reactive (`Flow.Publisher`) APIs for token-by-token streaming
- [x] **Bidirectional streaming** - Full support via `ModelStreamInfer` gRPC RPC for decoupled models
- [x] **Requested Output management** - `InferRequestedOutput` to filter returned outputs, reducing bandwidth and memory
- [x] **Secure communication (TLS)** - Plaintext, one-way TLS, and mutual TLS (mTLS) support

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