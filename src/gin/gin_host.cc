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

// Per-thread progress worker. Thread t owns GIN connections
// t, t+proxyNthreads, t+2*proxyNthreads, ... (round-robin across backends[0])
// for all devComms in ginState->devComms.
//
// The pause protocol ensures devComms list mutations (setup/free) are safe:
// all running threads transition Running -> PauseReq -> Paused before the
// mutation, then back to Running after.
//
// State machine (per thread, under mutex[t]):
//
//   ┌──────────┐  pauseAll   ┌──────────┐  ack (set Paused)  ┌────────┐
//   │ Running  │ ──────────> │ PauseReq │ ─────────────────> │ Paused │
//   └──────────┘             └──────────┘                    └────────┘
//        ^                  resumeAll (set Running)               │
//        └────────────────────────────────────────────────────────┘
//
//   Running  : traverse devComms, call ginProgress(). Mutex released during
//              plugin call so the main thread is not blocked.
//   PauseReq : main thread requested pause; worker acks by moving to Paused.
//   Paused   : worker sleeps on cond. Main thread mutates devComms list safely.
//   Exit     : clean shutdown; worker returns.
//   Error    : terminal; worker returns after storing asyncResult.
//
void* ncclGinProgress(struct ncclGinState* ginState, int t) {
  struct ncclGinBackendState* backend = &ginState->backends[0];
  if (ncclOsCpuCount(ginState->cpuAffinity)) {
    ncclOsSetAffinity(ginState->cpuAffinity);
  }
  while (1) {
    std::unique_lock<std::mutex> lock(ginState->threadMutex[t]);
    if (ginState->ginProgress[t] == ncclGinProgressRunning) {
      lock.unlock();
      struct ncclGinStateDevComm* dc = ginState->devComms;
      while (dc) {
        for (int n = t; n < backend->ginCommCount; n += ginState->proxyNthreads) {
          ncclResult_t ret = backend->ncclGin->ginProgress(dc->ginCtx[n]);
          if (ret != ncclSuccess) {
            COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
            INFO_LOC(NCCL_ALL, "-> %d [GIN Progress Thread %d]", ret, t);
            std::lock_guard<std::mutex> errLock(ginState->threadMutex[t]);
            ginState->ginProgress[t] = ncclGinProgressError;
            return NULL;
          }
        }
        dc = dc->next;
      }
      std::this_thread::yield();
    } else if (ginState->ginProgress[t] == ncclGinProgressPauseReq) {
      // Pause requested: acknowledge and sleep. The main thread is mutating the
      // devComms list (or freeing contexts) and will signal back to Running.
      ginState->ginProgress[t] = ncclGinProgressPaused;
      ginState->cond[t].notify_one();
      // No wait here — loop back, see Paused, and wait in Paused case below.
    } else if (ginState->ginProgress[t] == ncclGinProgressPaused) {
      ginState->cond[t].wait(lock);
    } else if (ginState->ginProgress[t] == ncclGinProgressExit) {
      return NULL;
    } else {
      INFO_LOC(NCCL_ALL, "[GIN Progress Thread %d] state unknown %d", t, ginState->ginProgress[t]);
      ginState->ginProgress[t] = ncclGinProgressError;
      return NULL;
    }
  }
}

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);
NCCL_PARAM(GinProxyNthreads, "GIN_PROXY_NTHREADS", 1);

// Request all running progress threads to pause (Running -> PauseReq ->
// Paused) and wait for each to acknowledge. Threads that are already
// Paused, Exiting, or Errored are left as-is. Used to safely mutate the
// devComms linkedlist or free GIN contexts while no progress thread is
// iterating.
static void pauseAllProgressThreads(struct ncclGinState* ginState) {
  for (int t = 0; t < ginState->proxyNthreads; t++) {
    std::unique_lock<std::mutex> lock(ginState->threadMutex[t]);
    if (ginState->ginProgress[t] == ncclGinProgressRunning) {
      ginState->ginProgress[t] = ncclGinProgressPauseReq;

      // Wake the worker if it's sleeping in Paused state from a prior cycle;
      // if it's in Running (common case), this notify is a harmless no-op and
      // the worker will see PauseReq on its next lock re-acquisition.
      ginState->cond[t].notify_one();
      // Wait (releases mutex) until worker acknowledges by leaving PauseReq state.
      ginState->cond[t].wait(lock, [&] { return ginState->ginProgress[t] != ncclGinProgressPauseReq; });
    }
  }
}

// Resume all paused progress threads (state Paused -> Running). Threads
// that are running, exiting, or errored are left as-is.
static void resumeAllProgressThreads(struct ncclGinState* ginState) {
  for (int t = 0; t < ginState->proxyNthreads; t++) {
    std::unique_lock<std::mutex> lock(ginState->threadMutex[t]);
    if (ginState->ginProgress[t] == ncclGinProgressPaused) {
      ginState->ginProgress[t] = ncclGinProgressRunning;
      ginState->cond[t].notify_one();
    }
  }
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
  INFO(NCCL_INIT, "GIN: %d connection(s) distributed across %d progress thread(s) (stride %d)",
       backend->ginCommCount, nthreads, nthreads);

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

// Serialized against concurrent Setup/Free via ginState->mutex.
// Progress threads are synchronized via pauseAllProgressThreads() /
// resumeAllProgressThreads().
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
    WARN("Cannot get backend version for unsupported GIN type %d", backend->ginType);
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

  // Add devComm context to the list and (re)start progress threads as needed.
  {
    std::unique_lock<std::mutex> lock(ginState->mutex);
    bool needsStart = needsProxyProgress && !ginState->proxyThreadsStarted;

    // If threads are already running, pause them so we can mutate devComms safely.
    if (ginState->proxyThreadsStarted) pauseAllProgressThreads(ginState);

    // Append the new devComm. Safe under one of:
    //  - threads not yet started (no concurrent reader), or
    //  - all threads paused via pauseAllProgressThreads().
    struct ncclGinStateDevComm* last = ginState->devComms;
    if (last) {
      while (last->next) last = last->next;
      last->next = ginStateDevComm;
    } else {
      ginState->devComms = ginStateDevComm;
    }

    if (needsStart) {
      // First-time start: spawn one progress thread per connection stride.
      ginState->cpuAffinity = comm->cpuAffinity;
      for (int t = 0; t < ginState->proxyNthreads; t++) {
        ginState->ginProgress[t] = ncclGinProgressRunning;
        ginState->thread[t] = std::thread([ginState, t] { ncclGinProgress(ginState, t); });
        ncclSetThreadName(ginState->thread[t], "NCCL GIN Progress%2d-%d", comm->cudaDev, t);
      }
      ginState->proxyThreadsStarted = true;
    } else if (ginState->proxyThreadsStarted) {
      // Threads were paused above; wake them now that the list is updated.
      resumeAllProgressThreads(ginState);
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

// Called from main thread; Same serialization as ncclGinDevCommSetup() via
// ginState->mutex.
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  // Find the resource associated with this devComm. Use the gin handle as key.
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinBackendState* backend = &ginState->backends[0];

  std::unique_lock<std::mutex> lock(ginState->mutex);

  // Pause progress threads before traversing and unlinking, so no worker
  // can be iterating the list concurrently.
  if (ginState->proxyThreadsStarted) pauseAllProgressThreads(ginState);

  // Locate the devComm. Use the gin handle as key.
  struct ncclGinStateDevComm *dc = ginState->devComms, *prevDc = NULL;
  while (1) {
    if (dc == NULL) {
      WARN("Dev comm not found\n");
      if (ginState->proxyThreadsStarted) resumeAllProgressThreads(ginState);
      return ncclInternalError;
    }
    if (dc->devHandles[0]->handle == devComm->ginHandles[0]) break;
    prevDc = dc;
    dc = dc->next;
  }

  // Remove from linked list. Workers are paused so the pointer update is safe.
  if (prevDc) prevDc->next = dc->next;
  else ginState->devComms = dc->next;
  if (ginState->proxyThreadsStarted) resumeAllProgressThreads(ginState);

  lock.unlock();

  // The devComm is now unreachable by any progress thread; safe to destroy
  // its contexts while the workers keep progressing the rest of the list.
  ncclResult_t ret = ncclSuccess;
  for (int n = 0; n < backend->ginCommCount; n++) {
    ncclResult_t r = backend->ncclGin->destroyContext(dc->ginCtx[n]);
    if (r != ncclSuccess && ret == ncclSuccess) ret = r;
  }
  free(dc);
  return ret;
}

ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (!ginState->connected) return ncclSuccess;
  struct ncclGinBackendState* backend = &ginState->backends[0];

  if (ginState->proxyThreadsStarted) {
    // Signal each progress thread to exit, then join. Errored threads
    // (ncclGinProgressError) may have already returned; thread.join() is still
    // valid as long as the std::thread is joinable, and a final Exit store
    // under the per-thread mutex is harmless.
    for (int t = 0; t < ginState->proxyNthreads; t++) {
      std::lock_guard<std::mutex> lock(ginState->threadMutex[t]);
      ginState->ginProgress[t] = ncclGinProgressExit;
      ginState->cond[t].notify_one();
    }
    for (int t = 0; t < ginState->proxyNthreads; t++) {
      if (ginState->thread[t].joinable()) ginState->thread[t].join();
    }
    ginState->proxyThreadsStarted = false;
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
