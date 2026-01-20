#include <cuda_runtime.h>
#include <cstdio>

struct ProfileCUDAGraph {
    cudaStream_t stream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaEvent_t start, stop;
    
    int graphRepeat;
    int preWarmup;
    int warmupLaunches;
    int timedLaunches;

    ProfileCUDAGraph(
        int graphRepeat = 1000,
        int preWarmup = 2,
        int warmupLaunches = 10,
        int timedLaunches = 10
    ) : graphRepeat(graphRepeat),
        preWarmup(preWarmup),
        warmupLaunches(warmupLaunches),
        timedLaunches(timedLaunches)
    {
        cudaStreamCreate(&stream);
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~ProfileCUDAGraph() {
        cleanup();
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
    }

    void cleanup() {
        if (graphExec) { cudaGraphExecDestroy(graphExec); graphExec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
    }

    template<typename Func, typename... Args>
    float profile(Func&& func, Args&&... args) {
        cleanup();
        
        // Pre-warmup
        for (int i = 0; i < preWarmup; i++)
            func(stream, std::forward<Args>(args)...);
        cudaStreamSynchronize(stream);
        
        // Capture
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < graphRepeat; i++)
            func(stream, std::forward<Args>(args)...);
        
        if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) {
            printf("Capture failed\n");
            return -1.0f;
        }
        
        if (cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) != cudaSuccess) {
            printf("Instantiate failed\n");
            return -1.0f;
        }
        
        // Warmup
        for (int i = 0; i < warmupLaunches; i++)
            cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        
        // Timed
        cudaEventRecord(start, stream);
        for (int i = 0; i < timedLaunches; i++)
            cudaGraphLaunch(graphExec, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms / (timedLaunches * graphRepeat);
    }
};