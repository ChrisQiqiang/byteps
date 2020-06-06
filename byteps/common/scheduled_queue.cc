// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "scheduled_queue.h"
#include <algorithm>
#include "global.h"
#include "logging.h"

namespace byteps {
namespace common {

BytePSScheduledQueue::BytePSScheduledQueue(QueueType type) {
  if ((type == REDUCE && BytePSGlobal::GetNccl()->IsSignalRoot()) || type == PUSH || type == PULL)  {
    _is_scheduled = true;
  } else {
    _is_scheduled = false;
  }

  size_t credit_in_partition = BytePSGlobal::GetNccl()->GetGroupSize() + 1;

  auto byteps_scheduling_credit = getenv("BYTEPS_SCHEDULING_CREDIT");
  credit_in_partition = byteps_scheduling_credit ? atoi(byteps_scheduling_credit) : 0;
  if (!credit_in_partition) { // disable scheduling by default
    _is_scheduled = false;
  }
  _qt = type;
  _credits = _is_scheduled
              ? BytePSGlobal::GetPartitionBound() * credit_in_partition
              : 34359738368;  // 32GB, basically disabling credit control

  // _is_scheduled = (_is_scheduled || _qt == PUSH || _qt == PULL)
  _rt = nullptr;
  auto _w_size = getenv("CHRIS_WINDOW_SIZE");
  _window_size = _w_size ? atoi(_w_size) : 4;
  switch (_qt) {
    case REDUCE:
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetReduceTable();
      }
      break;
    case PCIE_REDUCE:
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        if (BytePSGlobal::GetCpuReducer()->isRoot()) {
          _rt = BytePSGlobal::GetPcieReduceTable();
        }
      }
      break;
    case PUSH:
      if (BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetPushTable();
      }
      break;
    case PULL:
      if(BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetPullTable();
      }
    case COPYH2D:
      if (!BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetCopyTable();
      }
      break;
    case BROADCAST:
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetBroadcastTable();
      }
      break;
    default:
      break;
  }
}

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
  std::lock_guard<std::mutex> lock(_mutex);
  
  if (_is_scheduled) {
    // TODO: below can be optimized to O(n) using insertion sort
      bool flag = false;
      for(auto it = _sq.begin(); it != _sq.end(); it++){
        auto task = *it;
        if(task -> priority > entry -> priority || (task -> priority == entry -> priority && task -> key < entry -> key))
          continue;
        else 
        {
          flag = true;
          _sq.insert(it, entry);
          break;
        }
      }
      if(!flag)_sq.push_back(entry);
      //  std::sort(
      //   _sq.begin(), _sq.end(),
      //   [](std::shared_ptr<TensorTableEntry> a,
      //      std::shared_ptr<TensorTableEntry> b) {
      //     if (a->priority == b->priority) {
      //       return (a->key < b->key);  // from the first partition to the last
      //     }
      //     return (a->priority > b->priority);  // from higher priority to lower
      //   });
  }
  else
    _sq.push_back(entry);
  
  BPS_CHECK(entry->tensor_name != "");
  BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                 << " addTask: " << entry->tensor_name << " key: " << entry->key
                 << " rank: " << BytePSGlobal::GetLocalRank();
  return;
}

// Record the start time of the sub-tasks for all QueueTypes of each partition.
void BytePSScheduledQueue::recorderTs(std::shared_ptr<TensorTableEntry> task) {
  auto context = task->context;
  // add for profiling
  if (context->profile_flag) {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    auto &queue_list = task->queue_list;
    BPS_CHECK_GE(queue_list.size(), 1);
    auto this_op = queue_list[0];

    BPSCommTime *ret = new BPSCommTime;
    ret->start_t = (long long)(us.count());
    ret->key = task->key;
    ret->type = this_op;
    context->part_comm_time[task->key][this_op].push(ret);
  }
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<TensorTableEntry> task;
  // TODO: below can be optimized -- if we take task from the tail, erase() can
  // be faster
  for (auto it = _sq.begin(); it != _sq.end(); ++it) {
    if ((*it)->ready_event) {
      if (!(*it)->ready_event->Ready()) {
        continue;
      }
    }
    if (_is_scheduled) {
      if ((_qt == REDUCE && (*it)->len > _credits) || (_qt != REDUCE && _transfer_window.size() >= _window_size)) {
        continue;
      }
    }
    if (_rt) {
      if (!_rt->IsKeyReady((*it)->key)) {
        continue;
      }
      _rt->ClearReadyCount((*it)->key);
    }
    task = *it;
    _sq.erase(it);
    if (_is_scheduled) {
      if(_qt == REDUCE)_credits -= task->len;
      else
        _transfer_window.insert(task -> priority);
    }

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                  << " getTask: " << task->tensor_name << " key: " << task->key
                  << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    recorderTs(task);
    return task;
    

  }
  return nullptr;
}


std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask(uint64_t key) {
  BPS_CHECK(!_is_scheduled);
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<TensorTableEntry> task;
  for (auto it = _sq.begin(); it != _sq.end(); ++it) {
    if ((*it)->ready_event) {
      BPS_CHECK((*it)->ready_event->Ready());
    }
    if ((*it)->key != (uint64_t)key) {
      continue;
    }
    task = *it;
    _sq.erase(it);

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                   << " getTask(key): " << task->tensor_name
                   << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    recorderTs(task);
    return task;
  }
  return nullptr;
}

uint32_t BytePSScheduledQueue::pendingSize() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _sq.size();
}

void BytePSScheduledQueue::reportFinish(std::shared_ptr<TensorTableEntry> task) {
  if (_is_scheduled) {
    std::lock_guard<std::mutex> lock(_mutex);
    if(_qt == REDUCE)_credits += task -> len;
    else
      _transfer_window.erase(_transfer_window.find(task -> priority));
  }
  return;
}

int BytePSScheduledQueue::get_max_priority(){
    std::lock_guard<std::mutex> lock(_mutex);
    if(!_transfer_window.empty() && _sq.size()){
      auto first = _sq.begin();
      return std::max(*(_transfer_window.begin()), (*first) -> priority) ;
    }
    else if(!_transfer_window.empty()){
      return *(_transfer_window.begin());
    }
    else if(_sq.size()){
      auto first = _sq.begin();
      return (*first) -> priority;
    }
    else
      return 1; 
  }

int BytePSScheduledQueue::get_min_priority(){
    std::lock_guard<std::mutex> lock(_mutex);
    if(!_transfer_window.empty() && _sq.size()){
      auto first = _sq.begin();
      if(_transfer_window.size() < _window_size)
        return std::min(*(_transfer_window.rbegin()), (*first) -> priority);
      else
        return  *(_transfer_window.rbegin()); 
    }
    else if(!_transfer_window.empty()){
      return *(_transfer_window.rbegin());
    }
    else if(_sq.size()){
      auto first = _sq.begin();
      return (*first) -> priority;
    }
    else
      return 1; 
  }

int BytePSScheduledQueue::get_first_element(){
  std::lock_guard<std::mutex> lock(_mutex);
  if(!_sq.size())
    return 1;
  else{
    auto first = _sq.begin();
    return (*first) -> priority;
  }
}

int BytePSScheduledQueue::get_transfer_window_size(){
  std::lock_guard<std::mutex> lock(_mutex);
  // int res;
  // if(_transfer_window.empty())return 1;
  // if(maximal == 1){
  //   res = *(_transfer_window.begin());
  // }
  // else
  //   res = *(_transfer_window.rbegin());
  return _transfer_window.size();

}
}  // namespace common
}  // namespace byteps
