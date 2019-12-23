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
  if (type == REDUCE && BytePSGlobal::GetNccl()->IsSignalRoot()) {
    _is_scheduled = true;
  } else {
    _is_scheduled = false;
  }

  size_t credit_in_partition = BytePSGlobal::GetNccl()->GetGroupSize() + 1;
  if (getenv("BYTEPS_SCHEDULING_CREDIT")) {
    credit_in_partition = atoi(getenv("BYTEPS_SCHEDULING_CREDIT"));
  }
  if (!credit_in_partition) {
    _is_scheduled = false;
  }

  _qt = type;
  _credits = _is_scheduled
              ? BytePSGlobal::GetPartitionBound() * credit_in_partition
              : 34359738368;  // 32GB, basically disabling credit control
  _rt = nullptr;

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
      //BPS_LOG(DEBUG) << "IN PUSH: " << _is_scheduled ;
      if (BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetPushTable();
      }
      _tensor_num=0;
      for(int i = 11; i >= 0; i--)
      {
        for(int j = _grad_checkpoint[i + 1] - 1; j > _middle[i]; j--){
            _mystack.push(j * -1 );
            BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
        }
      }
      for(int i = 0 ; i <= 11; i++)
      {
        for(int j = _middle[i] ; j >= _grad_checkpoint[i]; j--){
            _mystack.push(j * -1);
            BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
        }
      }
      for(int i = 0;i < 160; i++)
      {
        // _vis[i] = 0;
        _tensor_part[i] = 0;
      }
      BPS_LOG(DEBUG) << " Done. DOOR IS " << _dooropen ;
      break;
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
  _sq.push_back(entry);
  if (_is_scheduled) {
    // TODO: below can be optimized to O(n) using insertion sort
    std::sort(
        _sq.begin(), _sq.end(),
        [](std::shared_ptr<TensorTableEntry> a,
           std::shared_ptr<TensorTableEntry> b) {
          if (a->priority == b->priority) {
            return (a->key < b->key);  // from the first partition to the last
          }
          return (a->priority > b->priority);  // from higher priority to lower
        });
  }
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
      if ((*it)->len > _credits)
        continue;
    }
    if (_rt) {
      if (!_rt->IsKeyReady((*it)->key)) {
        continue;
      }
      _rt->ClearReadyCount((*it)->key);
    }
    std::string tmp = (*it) -> tensor_name;
    task = *it;
    BPS_LOG(DEBUG) << _qt << " tensor name: " << tmp;
    if(_qt == PUSH && tmp.find("gradient") != tmp.npos )
    {
        BPS_LOG(DEBUG) << "Task: " <<  task-> priority << "I have meet zero: " << _meetzero << " and door is open: " << _dooropen;
        if(task -> priority == 0) {
          _meetzero = 1;
         BPS_LOG(DEBUG) << "Meet zero.";
         }
        if(!_meetzero)
        {
            if(task -> priority !=  _mystack.top())continue; 
            BPS_LOG(INFO) << "PUSH GRADIENT: " << tmp;
            _tensor_part[ task -> priority * -1]++; 
            if(_tensor_part[task -> priority * -1 ] == 1 && task -> total_partnum > 1){
              for(int base = 1; base < task-> total_partnum ; base++)
                _mystack.push(task -> priority);//the values in the stack and priority are both negative
                BPS_LOG(INFO) << "PUSH elements into mystack  IN THE PROCESS: " << tmp;
            }
            if(_tensor_part[ task -> priority * -1 ] == task -> total_partnum )_tensor_num++;
            _mystack.pop();
        }
        else if(!_dooropen) {//we cannot change the value of tensor_part if door is closed.
          BPS_LOG(DEBUG) << "door is closed.";
          break;
        }
        else {
           BPS_LOG(DEBUG) << "Tensor name: " << tmp << "   myqueue top: " << _mystack.top()  << "  size of _sq: " << _sq.size();    
           if(task -> priority !=  _mystack.top())continue; 
           BPS_LOG(INFO) << "PUSH GRADIENT: " << tmp;
           BPS_LOG(DEBUG) << "Pass, and dooopen --";
            _tensor_part[ task -> priority * -1]++; 
            if(_tensor_part[task -> priority * -1 ] == 1 && task -> total_partnum > 1){
              for(int base = 1; base < task-> total_partnum ; base++)
                _mystack.push(task -> priority);
                BPS_LOG(INFO) << "PUSH elements into mystack IN THE PROCESS: " << tmp;
            }
            if(_tensor_part[ task -> priority * -1 ] == task -> total_partnum )_tensor_num++;
            _mystack.pop();
            _dooropen--;
            BPS_LOG(DEBUG) << "PUSH gradient: " << tmp ;
            // BPS_LOG(DEBUG) << "The door has been closed.";
        }
         BPS_LOG(DEBUG) << "transferred tensor num: " << _tensor_num  << "  empty: " << _mystack.empty() << " size of myqueue: " << _mystack.size();

        //all push process end in this iteration , then reinitalize varibles.
        if(_tensor_num == 157 && _mystack.empty())
        {
          BPS_LOG(DEBUG) << "Clear.";
          _meetzero = 0;
          _dooropen = 11;
          // _doorcount = 0;
          _tensor_num = 0;
          for(int i = 0; i < 160; i++)_tensor_part[i] = 0;
          // for(int i = 0;i < 160; i++) _vis[i] = 0;  
          for(int i = 11; i >= 0; i--)
          {
            for(int j = _grad_checkpoint[i + 1] - 1; j > _middle[i]; j--){
                _mystack.push(j * -1 );
                BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
            }
          }
          for(int i = 0 ; i <= 11; i++)
          {
            for(int j = _middle[i] ; j >= _grad_checkpoint[i]; j--){
                _mystack.push(j * -1);
                BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
            }
          }
        }
    }    
    if (_is_scheduled) 
    {
        _credits -= task->len;
    }
    _sq.erase(it);
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

void BytePSScheduledQueue::reportFinish(int size) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_is_scheduled) {
      _credits += size;
  }
  if(_qt == PUSH)
  {
    if(_meetzero) {
         if(_dooropen < 11)
              _dooropen++;
         }       
         // BPS_LOG(DEBUG) << "door open value:" << _dooropen;
  }
  return;
}

}  // namespace common
}  // namespace byteps
