/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "gin.h"
#include "param.h"
#include "graph.h"
#include "transport.h"
#include "register_inline.h"
#include "gin/gin_host.h"
#include "gin/gin_host_proxy.h"
#include "compiler.h"
#include <cmath>

NCCL_PARAM(GinEnable, "GIN_ENABLE", 1);
NCCL_PARAM(DevApiJit, "DEV_API_JIT", 0);

// Backend version compatibility. Index: backend version. Value: min compatible NCCL version
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};

ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType = comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL ? NCCL_GIN_TYPE_NONE :
                                                                  comm->sharedRes->ginState.backends[0].ginType;
  return ncclSuccess;
}

ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType = comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE ? NCCL_GIN_TYPE_NONE :
                                                                  comm->sharedRes->ginState.backends[0].ginType;
  return ncclSuccess;
}

// Legacy progress worker (NCCL_GIN_PROXY_NTHREADS < 1): original upstream behavior.
// Single thread, single mutex, checks proxyThreadStopSignal each iteration.
static void* ncclGinLegacyProgress(struct ncclGinState* ginState) {
  struct ncclGinBackendState* backend = &ginState->backends[0];
  if (ncclOsCpuCount(ginState->cpuAffinity)) {
    ncclOsSetAffinity(ginState->cpuAffinity);
  }
  while (1) {
    std::unique_lock<std::mutex> lock(ginState->mutex);
    if (ginState->proxyThreadStopSignal) return NULL;
    struct ncclGinStateDevComm* dc = ginState->devComms;
    while (dc) {
      for (int n = 0; n < backend->ginCommCount; n++) {
        ncclResult_t ret = backend->ncclGin->ginProgress(dc->ginCtx[n]);
        if (ret != ncclSuccess) {
          COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
          INFO_LOC(NCCL_ALL, "-> %d [GIN Progress Thread]", ret);
          return NULL;
        }
      }
      dc = dc->next;
    }
    lock.unlock();
    std::this_thread::yield();
  }
}

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);
NCCL_PARAM(GinProxyNthreads, "GIN_PROXY_NTHREADS", -1);

// Per-device global GIN progress thread pool. Shared across all comms on the
// same cudaDev. Threads start Paused and are resumed when ginCtx is assigned.
// When NCCL_GIN_PROXY_NTHREADS < 1 (default -1), the pool is not used and
// the legacy per-comm single thread is spawned instead.
struct ncclGinThreadPool {
  std::mutex poolMutex;
  int nthreads = 0;
  bool started = false;
  int refCount = 0;
  ncclAffinity cpuAffinity;

  std::thread thread[NCCL_GIN_MAX_CONNECTIONS];
  std::mutex threadMutex[NCCL_GIN_MAX_CONNECTIONS];
  std::condition_variable cond[NCCL_GIN_MAX_CONNECTIONS];
  int state[NCCL_GIN_MAX_CONNECTIONS] = {};  // ncclGinProgressState

  struct WorkItem {
    ncclGinState* ginState;
    void* ginCtx;
    ncclGin_t* ncclGin;
    bool disabled;  // Set on error; skipped on subsequent iterations.
  };
  std::vector<WorkItem> workList[NCCL_GIN_MAX_CONNECTIONS];
};
static ncclGinThreadPool g_progressPool[NCCL_MAX_LOCAL_RANKS];

// Per-device pool progress worker. Thread t on device dev progresses all
// ginCtx registered in pool.workList[t] via round-robin.
//
// The pause protocol ensures devComms list mutations (setup/free) are safe:
// all running threads transition Running -> PauseReq -> Paused before the
// mutation, then back to Running after.
//
// State machine (per thread, under pool.threadMutex[t]):
//
//   ┌──────────┐  pauseThreadPool  ┌──────────┐  ack (set Paused) ┌────────┐
//   │ Running  │ ────────────────> │ PauseReq │ ────────────────> │ Paused │
//   └──────────┘                   └──────────┘                   └────────┘
//        ^             resumeThreadPool (set Running)                 │
//        └────────────────────────────────────────────────────────────┘
//
//   Running  : iterate workList[t], call ginProgress(). Mutex released during
//              plugin calls so pauseThreadPool is not blocked.
//   PauseReq : main thread requested pause; worker acks by moving to Paused.
//   Paused   : worker sleeps on cond. Main thread mutates workList safely.
//              Initial state — no CPU cost until ginCtx is assigned.
//   Exit     : clean shutdown; worker returns.
//
//   On error from ginProgress(), the affected workItem is disabled (skipped on
//   subsequent iterations) and asyncResult is stored. The thread continues
//   progressing other workItems.
//
static void* ncclGinPoolProgress(int dev, int t) {
  auto& pool = g_progressPool[dev];
  if (ncclOsCpuCount(pool.cpuAffinity)) {
    ncclOsSetAffinity(pool.cpuAffinity);
  }
  while (true) {
    std::unique_lock<std::mutex> lock(pool.threadMutex[t]);
    if (pool.state[t] == ncclGinProgressRunning) {
      lock.unlock();
      for (auto& item : pool.workList[t]) {
        if (item.disabled) continue;
        ncclResult_t ret = item.ncclGin->ginProgress(item.ginCtx);
        if (ret != ncclSuccess) {
          COMPILER_ATOMIC_STORE(&item.ginState->asyncResult, ret, std::memory_order_release);
          INFO_LOC(NCCL_ALL, "-> %d [GIN Pool Progress %d-%d]", ret, dev, t);
          item.disabled = true;
        }
      }
      std::this_thread::yield();
    } else if (pool.state[t] == ncclGinProgressPauseReq) {
      pool.state[t] = ncclGinProgressPaused;
      pool.cond[t].notify_one();
    } else if (pool.state[t] == ncclGinProgressPaused) {
      pool.cond[t].wait(lock);
    } else if (pool.state[t] == ncclGinProgressExit) {
      return NULL;
    }
  }
}

// Pause all running pool threads on the given device. Blocks until each
// acknowledges (Running -> PauseReq -> Paused). Safe to mutate workLists after.
static void pauseThreadPool(int dev) {
  auto& pool = g_progressPool[dev];
  for (int t = 0; t < pool.nthreads; t++) {
    std::unique_lock<std::mutex> lock(pool.threadMutex[t]);
    if (pool.state[t] == ncclGinProgressRunning) {
      pool.state[t] = ncclGinProgressPauseReq;
      pool.cond[t].notify_one();
      pool.cond[t].wait(lock, [&] { return pool.state[t] != ncclGinProgressPauseReq; });
    }
  }
}

// Resume paused pool threads that have work. Threads with empty workLists
// remain Paused (zero CPU cost).
static void resumeThreadPool(int dev) {
  auto& pool = g_progressPool[dev];
  for (int t = 0; t < pool.nthreads; t++) {
    std::unique_lock<std::mutex> lock(pool.threadMutex[t]);
    if (pool.state[t] == ncclGinProgressPaused && !pool.workList[t].empty()) {
      pool.state[t] = ncclGinProgressRunning;
      pool.cond[t].notify_one();
    }
  }
}

// Initialize the per-device thread pool. Idempotent: returns immediately if
// already started. Threads start in Paused state (no CPU cost until resumed).
static void initThreadPool(int dev, int nthreads, ncclAffinity cpuAffinity) {
  auto& pool = g_progressPool[dev];
  std::lock_guard<std::mutex> lock(pool.poolMutex);
  if (pool.started) return;
  pool.nthreads = nthreads;
  pool.cpuAffinity = cpuAffinity;
  for (int t = 0; t < nthreads; t++) {
    pool.state[t] = ncclGinProgressPaused;
    pool.thread[t] = std::thread(ncclGinPoolProgress, dev, t);
  }
  pool.started = true;
}

static bool useGlobalThreadPool() {
  return ncclParamGinProxyNthreads() >= 1;
}

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  ncclTeam_t ginTeam;
  if (ginState->connected) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  if (ncclParamGinEnable() == 0) {
    WARN("GIN is disabled.");
    return ncclInternalError;
  }

  if (!ginState->supported) {
    WARN("GIN not supported.");
    return ncclInvalidUsage;
  }
  struct ncclGinBackendState* backend = &ginState->backends[0];

  ginState->ginConnectionType = comm->globalGinSupport;

  int ndev = 0;
  NCCLCHECK(backend->ncclGin->devices(&ndev));
  if (ndev <= 0) {
    WARN("No GIN-capable devices found.");
    return ncclInternalError;
  }

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    return ncclInternalError;
  }

  int nLocalGinDevs;
  int localGinDevs[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalGinDevs(comm, localGinDevs, &nLocalGinDevs));

  void** handles = NULL;
  char* allHandles = NULL;
  void* listenComm = NULL;

  int* ginCommCountHandles = NULL;
  NCCLCHECKGOTO(ncclCalloc(&ginCommCountHandles, comm->nRanks), ret, fail);

  backend->ginCommCount = nLocalGinDevs;
  if (backend->ginVersion < 13) {
    // We only support one context per connection, so we better create as many connections as possible.
    backend->ginCommCount = NCCL_GIN_MAX_CONNECTIONS;
  }

  // Resolve the number of GIN progress threads. Default 1; clamp to the valid
  // range [1, NCCL_GIN_MAX_CONNECTIONS]. nthreads is a purely local concern
  // (workers need not match across ranks), so it can be clamped per-rank.
  int nthreads;
  nthreads = (int)ncclParamGinProxyNthreads();
  if (nthreads < 1) {
    nthreads = 1;
  } else if (nthreads > NCCL_GIN_MAX_CONNECTIONS) {
    WARN("GIN_PROXY_NTHREADS=%d exceeds the maximum number of connections %d; clamping to %d", nthreads,
         NCCL_GIN_MAX_CONNECTIONS, NCCL_GIN_MAX_CONNECTIONS);
    nthreads = NCCL_GIN_MAX_CONNECTIONS;
  }

  if (ncclParamGinNconnections() != -2) backend->ginCommCount = ncclParamGinNconnections();
  backend->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, backend->ginCommCount);

  ginCommCountHandles[comm->rank] = backend->ginCommCount;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) {
    backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
  }

  // Now that the final connection count is known (after the cross-rank min),
  // clamp the thread count to it: we never spawn empty-range threads.
  if (nthreads > backend->ginCommCount) {
    INFO(NCCL_INIT, "GIN: clamping GIN_PROXY_NTHREADS from %d to %d (resolved ginCommCount)", nthreads,
         backend->ginCommCount);
    nthreads = backend->ginCommCount;
  }
  ginState->proxyNthreads = nthreads;
  ginState->progressMode = useGlobalThreadPool() ? ncclGinProgressPool : ncclGinProgressLegacy;
  INFO(NCCL_INIT, "GIN: %d connection(s), %d progress thread(s), mode=%s",
       backend->ginCommCount, nthreads,
       ginState->progressMode == ncclGinProgressPool ? "shared thread pool" : "thread per comm");

  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);

  // Connect the maximum supported connection type. Any future devComm may request
  // up to this connection type.
  ginTeam = ncclTeamWorld(comm);
  if (ginState->ginConnectionType != NCCL_GIN_CONNECTION_FULL) {
    ginTeam = {
      .nRanks = comm->nRanks / comm->contiguousRanksPerHost,
      .rank = comm->rank / comm->contiguousRanksPerHost,
      .stride = comm->contiguousRanksPerHost,
    };
  }
  for (int r = 0; r < ginTeam.nRanks; r++) {
    int worldRank = ncclTeamRankToWorld(comm, ginTeam, r);
    handles[r] = allHandles + worldRank * NCCL_NET_HANDLE_MAXSIZE;
  }

  for (int n = 0; n < backend->ginCommCount; n++) {
    if (backend->ncclGin->setHint) {
      NCCLCHECKGOTO(backend->ncclGin->setHint(backend->ginInstance, "THREAD_IDX", n % nthreads), ret, fail);
    }
    NCCLCHECKGOTO(backend->ncclGin->listen(backend->ginInstance, localGinDevs[n % nLocalGinDevs],
                                           allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
                  ret, fail);

    NCCLCHECKGOTO(backend->ncclGin->getProperties(localGinDevs[n % nLocalGinDevs], backend->ginProps + n), ret, fail);

    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret, fail);

    NCCLCHECKGOTO(backend->ncclGin->connect(backend->ginInstance, handles, ginTeam.nRanks, ginTeam.rank, listenComm,
                                            backend->ginComms + n),
                  ret, fail);

    NCCLCHECKGOTO(backend->ncclGin->closeListen(listenComm), ret, fail);
    listenComm = NULL;
  }

exit:
  free(handles);
  free(allHandles);
  free(ginCommCountHandles);
  if (ret == ncclSuccess) ginState->connected = true;
  return ret;
fail:
  if (listenComm != NULL) NCCLCHECKIGNORE(backend->ncclGin->closeListen(listenComm), ret);
  for (int n = 0; n < backend->ginCommCount; n++) {
    if (backend->ginComms[n] != NULL) {
      NCCLCHECKIGNORE(backend->ncclGin->closeColl(backend->ginComms[n]), ret);
      backend->ginComms[n] = NULL;
    }
  }
  goto exit;
}

// Called from main thread; Setup and Free are never concurrent for the
// same comm (serialized by the caller). Progress threads are synchronized
// via pauseAllProgressThreads() / resumeAllProgressThreads().
ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinBackendState* backend = &ginState->backends[0];
  ncclGinConfig_t ginConfig;

  if (reqs->ginStrongSignalsRequired && !backend->supportsStrongSignals) {
    WARN("GIN strong signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  if (reqs->ginVaSignalsRequired && !backend->supportsVASignals) {
    WARN("GIN VA signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  devComm->ginSignalCount = reqs->ginSignalCount;
  devComm->ginCounterCount = reqs->ginCounterCount;
  // Legacy signals default to what is specified in DevCommRequirements
  devComm->ginStrongLegacySignals = reqs->ginStrongSignalsRequired;

  // Allocate contexts
  int nContextsTotal = reqs->ginContextCount;
  if (backend->ginVersion < 13) {
    nContextsTotal = backend->ginCommCount;
  }
  devComm->ginContextCount = nContextsTotal;
  devComm->ginConnectionCount = backend->ginCommCount;
  if (!reqs->ginExclusiveContexts) {
    // TODO: check if a shared devComm in the list could match our requirements.
  }

  nContextsTotal = ROUNDUP(nContextsTotal, backend->ginCommCount);
  int nContextsPerComm = nContextsTotal / backend->ginCommCount;
  INFO(NCCL_INIT,
       "devCommCreate: creating %d contexts: %d GIN connections with %d contexts each (%d contexts total requested)",
       nContextsTotal, backend->ginCommCount, nContextsPerComm, reqs->ginContextCount);

  struct ncclGinStateDevComm* ginStateDevComm = NULL;
  NCCLCHECK(ncclCalloc(&ginStateDevComm, 1));
  ginStateDevComm->contextCount = nContextsTotal;

  const int* backendVersionArray;
  int nVersions;
  switch (backend->ginType) {
  case NCCL_GIN_TYPE_PROXY:
    backendVersionArray = proxyBackendMinVersions;
    nVersions = sizeof(proxyBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_GDAKI:
    backendVersionArray = gdakiBackendMinVersions;
    nVersions = sizeof(gdakiBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_GPI:
    backendVersionArray = gpiBackendMinVersions;
    nVersions = sizeof(gpiBackendMinVersions) / sizeof(int);
    break;
  default:
    WARN("Cannot get backend version for invalid GIN type %d", backend->ginType);
    return ncclInternalError;
  }

  int backendVersion = 0;
  if (ncclParamDevApiJit() == 1) {
    // JIT: device code version is the latest version.
    backendVersion = nVersions - 1;
  } else {
    // Non-JIT: device code version matches reqs->version.
    for (int i = 0; i < nVersions; i++) {
      if (reqs->version >= backendVersionArray[i]) backendVersion = i;
      else break;
    }
  }

  ncclResult_t ret = ncclSuccess;
  bool needsProxyProgress = false;

  int connectedStride =
    comm->sharedRes->ginState.ginConnectionType == NCCL_GIN_CONNECTION_FULL ? 1 : comm->contiguousRanksPerHost;
  int requestedStride = 1;
  if (reqs->ginConnectionType == NCCL_GIN_CONNECTION_CUSTOM_STRIDE) {
    requestedStride = reqs->ginCustomStride;
  } else if (reqs->ginConnectionType == NCCL_GIN_CONNECTION_RAIL) {
    requestedStride = ncclTeamRail(comm).stride;
  }

  if (requestedStride == 0) {
    WARN("Cannot create DevComm with a GIN rank stride of 0. To disable GIN, set reqs->ginConnectionType to "
         "NCCL_GIN_CONNECTION_NONE.");
    ret = ncclInvalidUsage;
    goto end;
  }
  if (requestedStride > ncclTeamRail(comm).stride) {
    // Hierarchical barriers assume GIN is at least RAIL connected.
    WARN("Cannot create DevComm with a GIN rank stride %d greater than the rail team stride %d", requestedStride,
         ncclTeamRail(comm).stride);
    ret = ncclInvalidUsage;
    goto end;
  }
  if (requestedStride % connectedStride != 0) {
    WARN("Cannot create DevComm with the requested GIN rank stride %d, this comm only supports strides that are "
         "multiples of %d",
         requestedStride, connectedStride);
    ret = ncclInvalidUsage;
    goto end;
  }

  devComm->ginConnectionStride = connectedStride;
  devComm->ginConnectionStride_rcp32 = idivRcp32(connectedStride);
  devComm->ginContextStride = requestedStride;
  ginConfig = {
    reqs->ginSignalCount,
    reqs->ginCounterCount,
    nContextsPerComm,
    reqs->ginQueueDepth,
    reqs->ginTrafficClass != NCCL_CONFIG_UNDEF_INT ? reqs->ginTrafficClass : comm->config.trafficClass,
    backendVersion,
    /*rankStride*/ requestedStride / connectedStride,
  };

  for (int n = 0; n < backend->ginCommCount; n++) {
    NCCLCHECKGOTO(backend->ncclGin->createContext(backend->ginComms[n], &ginConfig, &ginStateDevComm->ginCtx[n],
                                                  &ginStateDevComm->devHandles[n]),
                  ret, end);
    if (ginStateDevComm->ginCtx[n] == NULL || ginStateDevComm->devHandles[n] == NULL ||
        ginStateDevComm->devHandles[n]->handle == NULL) {
      WARN("GIN plugin %s returned invalid context for connection %d: ginCtx=%p devHandle=%p handle=%p",
           backend->ncclGin->name, n, ginStateDevComm->ginCtx[n], ginStateDevComm->devHandles[n],
           ginStateDevComm->devHandles[n] ? ginStateDevComm->devHandles[n]->handle : NULL);
      ret = ncclInternalError;
      goto end;
    }
    devComm->ginNetDeviceTypes[n] = ginStateDevComm->devHandles[n]->netDeviceType;
    devComm->ginHandles[n] = ginStateDevComm->devHandles[n]->handle;

    if (ginStateDevComm->devHandles[n]->needsProxyProgress) needsProxyProgress = true;
  }

  // Add devComm context to the list and start/register progress as needed.
  if (ginState->progressMode == ncclGinProgressPool) {
    // Pool mode: register ginCtx with per-device shared thread pool.
    int dev = comm->cudaDev;
    initThreadPool(dev, ginState->proxyNthreads, comm->cpuAffinity);
    auto& pool = g_progressPool[dev];
    std::lock_guard<std::mutex> plock(pool.poolMutex);

    pauseThreadPool(dev);

    struct ncclGinStateDevComm* last = ginState->devComms;
    if (last) {
      while (last->next) last = last->next;
      last->next = ginStateDevComm;
    } else {
      ginState->devComms = ginStateDevComm;
    }

    for (int n = 0; n < backend->ginCommCount; n++) {
      int t = n % pool.nthreads;
      pool.workList[t].push_back({ginState, ginStateDevComm->ginCtx[n], backend->ncclGin, false});
      INFO(NCCL_INIT, "GIN pool: dev=%d thread=%d registered ginCtx[%d]=%p (ginState=%p)",
           dev, t, n, ginStateDevComm->ginCtx[n], ginState);
    }
    pool.refCount++;
    ginState->progressStarted = true;

    resumeThreadPool(dev);
  } else {
    // Legacy mode: single thread, single mutex.
    {
      std::unique_lock<std::mutex> lock(ginState->mutex);
      struct ncclGinStateDevComm* last = ginState->devComms;
      if (last) {
        while (last->next) last = last->next;
        last->next = ginStateDevComm;
      } else {
        ginState->devComms = ginStateDevComm;
      }
    }
    if (needsProxyProgress && !ginState->progressStarted) {
      ginState->cpuAffinity = comm->cpuAffinity;
      ginState->progressStarted = true;
      ginState->thread = std::thread(ncclGinLegacyProgress, ginState);
      ncclSetThreadName(ginState->thread, "NCCL GIN Progress%2d", comm->cudaDev);
    }
  }

end:
  if (ret != ncclSuccess) {
    for (int n = 0; n < backend->ginCommCount; n++) {
      if (ginStateDevComm->ginCtx[n]) backend->ncclGin->destroyContext(ginStateDevComm->ginCtx[n]);
    }
    devComm->ginContextCount = 0;
    free(ginStateDevComm);
  }
  return ret;
}

// Called from main thread; Same serialization assumption as ncclGinDevCommSetup()
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinBackendState* backend = &ginState->backends[0];

  if (ginState->progressMode == ncclGinProgressPool) {
    int dev = comm->cudaDev;
    auto& pool = g_progressPool[dev];
    std::lock_guard<std::mutex> plock(pool.poolMutex);
    pauseThreadPool(dev);

    // Locate and unlink devComm
    struct ncclGinStateDevComm *dc = ginState->devComms, *prevDc = NULL;
    while (dc && dc->devHandles[0]->handle != devComm->ginHandles[0]) {
      prevDc = dc; dc = dc->next;
    }
    if (!dc) { resumeThreadPool(dev); WARN("Dev comm not found"); return ncclInternalError; }
    if (prevDc) prevDc->next = dc->next; else ginState->devComms = dc->next;

    // Remove ginCtx entries from pool workLists
    for (int t = 0; t < pool.nthreads; t++) {
      auto& list = pool.workList[t];
      list.erase(std::remove_if(list.begin(), list.end(),
        [&](const ncclGinThreadPool::WorkItem& item) {
          for (int n = 0; n < backend->ginCommCount; n++) {
            if (item.ginCtx == dc->ginCtx[n]) return true;
          }
          return false;
        }), list.end());
    }

    resumeThreadPool(dev);

    // Destroy contexts (safe: no thread references them now)
    ncclResult_t ret = ncclSuccess;
    for (int n = 0; n < backend->ginCommCount; n++) {
      ncclResult_t r = backend->ncclGin->destroyContext(dc->ginCtx[n]);
      if (r != ncclSuccess && ret == ncclSuccess) ret = r;
    }
    free(dc);
    return ret;
  } else {
    // Legacy mode
    struct ncclGinStateDevComm *dc = ginState->devComms, *prevDc = NULL;
    while (1) {
      if (dc == NULL) { WARN("Dev comm not found"); return ncclInternalError; }
      if (dc->devHandles[0]->handle == devComm->ginHandles[0]) break;
      prevDc = dc; dc = dc->next;
    }
    std::unique_lock<std::mutex> lock(ginState->mutex);
    if (prevDc) prevDc->next = dc->next; else ginState->devComms = dc->next;
    lock.unlock();

    for (int n = 0; n < backend->ginCommCount; n++) {
      NCCLCHECK(backend->ncclGin->destroyContext(dc->ginCtx[n]));
    }
    free(dc);
    return ncclSuccess;
  }
}

ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (!ginState->connected) return ncclSuccess;
  struct ncclGinBackendState* backend = &ginState->backends[0];

  if (ginState->progressMode == ncclGinProgressPool) {
    int dev = comm->cudaDev;
    auto& pool = g_progressPool[dev];
    std::lock_guard<std::mutex> plock(pool.poolMutex);
    pool.refCount--;
    if (pool.refCount == 0) {
      for (int t = 0; t < pool.nthreads; t++) {
        std::lock_guard<std::mutex> tlock(pool.threadMutex[t]);
        pool.state[t] = ncclGinProgressExit;
        pool.cond[t].notify_one();
      }
      for (int t = 0; t < pool.nthreads; t++) {
        if (pool.thread[t].joinable()) pool.thread[t].join();
      }
      pool.started = false;
    }
  } else {
    // Legacy mode
    if (ginState->progressStarted) {
      {
        std::lock_guard<std::mutex> lock(ginState->mutex);
        ginState->proxyThreadStopSignal = true;
      }
      ginState->thread.join();
    }
  }

  for (int n = 0; n < backend->ginCommCount; n++) {
    if (backend->ginComms[n] != NULL) {
      NCCLCHECK(backend->ncclGin->closeColl(backend->ginComms[n]));
      backend->ginComms[n] = NULL;
    }
  }
  memset((void*)ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags, bool multiSegment,
                             int memType) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinBackendState* backend = &ginState->backends[0];
  if (multiSegment) {
    // Multi-segment GIN registration requires DMABUF support on all GIN connections
    for (int n = 0; n < backend->ginCommCount; n++) {
      if (!(backend->ginProps[n].ptrSupport & NCCL_PTR_DMABUF)) {
        WARN("Window registration of addresses that span multiple physical segments requires DMABUF support with GIN.");
        return ncclInvalidArgument;
      }
    }
  }
  int mrFlags = (winFlags & NCCL_WIN_STRICT_ORDERING) ? NCCL_NET_MR_FLAG_FORCE_SO : 0;
  for (int n = 0; n < backend->ginCommCount; n++) {
    NCCLCHECK(backend->ncclGin->regMrSym(backend->ginComms[n], address, size, memType, mrFlags, &ginHostWins[n],
                                         &ginDevWins[n]));
    if (ginHostWins[n] == NULL) {
      WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinBackendState* backend = &ginState->backends[0];
  for (int n = 0; n < backend->ginCommCount; n++) {
    NCCLCHECK(backend->ncclGin->deregMrSym(backend->ginComms[n], ginHostWins[n]));
  }
  return ncclSuccess;
}

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  *hasError = false;
  struct ncclGinBackendState* backend = &ginState->backends[0];
  struct ncclGinStateDevComm* dc = ginState->devComms;
  while (dc) {
    for (int n = 0; n < backend->ginCommCount; n++) {
      NCCLCHECK(backend->ncclGin->queryLastError(dc->ginCtx[n], hasError));
      if (*hasError) return ncclSuccess;
    }
    dc = dc->next;
  }
  return ncclSuccess;
}
