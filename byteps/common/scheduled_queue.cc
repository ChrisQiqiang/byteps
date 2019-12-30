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
#include <cmath>
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

    case PULL:
      if (BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetPullTable();
      }
      // dynamic_size = _backward_exec[_sizepointer + 1];
      _sizepointer=1;
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
  BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
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
  //  BPS_LOG(DEBUG) << _qt << " tensor name: " << tmp;

    if( _qt == PUSH && tmp.find("gradient") != tmp.npos ) 
    {
      
          /////first  enqueue as the gradient block coming, then dequeue dynamically.
        if(_dequeue != 1){
        //   BPS_LOG(DEBUG) << "Position 1" << " pointer: " <<  _pointer <<" stagestart: " << _stagestart << " mystack empty:" <<  _mystack.empty() \
        //         << "task name: " << task -> tensor_name ; 
 
          bool taskisstart = task -> priority == -1 * _grad_checkpoint[_pointer]  && _stagestart ;
          bool taskisproc = !_mystack.empty() && task -> priority > -1 * _grad_checkpoint[_pointer] \ 
                    && task -> priority  < -1 * _grad_checkpoint[_pointer - 1] \
                    && task -> priority == _mystack.top() + 1;
          bool starttagged = _stagestart && _tensor_part[_grad_checkpoint[_pointer]] ;
          bool proctagged = !_mystack.empty() && _tensor_part[(_mystack.top() + 1) * -1] \
                    && _mystack.top() + 1 > -1 * _grad_checkpoint[_pointer] \ 
                    && _mystack.top() + 1  < -1 * _grad_checkpoint[_pointer - 1];                                    ;
          if(!_mystack.empty() && _mystack.top() == -156) 
            BPS_LOG(DEBUG) << "proctagged when top is -156: " << proctagged; 
          if( taskisstart || taskisproc || starttagged || proctagged)
          {
            if(starttagged)
              for(int x = 0; x < _tensor_part[_grad_checkpoint[_pointer]]; x++){
                _mystack.push(_grad_checkpoint[_pointer] * -1);
                _stagestart = 0;
                BPS_LOG(DEBUG) << "ENQUEUE at start element not firstly: " << _grad_checkpoint[_pointer] * -1 << " mystack size: " << _mystack.size() ;
              }
            
            else if(proctagged){
              int tmp = _mystack.top() + 1;
              for(int x = 0; x < _tensor_part[tmp * -1]; x++){
                _mystack.push(tmp);
                BPS_LOG(DEBUG) << "ENQUEUE in proc element not firstly: " << tmp  << " mystack size: " << _mystack.size();
              }
            }

            else {
              if(taskisstart) _stagestart = 0; 
              _tensor_part[task -> priority * -1] = task -> total_partnum;
              for(int x = 0; x< task -> total_partnum; x++){
                _mystack.push(task -> priority);
                BPS_LOG(DEBUG) << "ENQUEUE element firstly: " << task -> priority ;
              }
            }
            if(!_mystack.empty() &&  _mystack.top() * -1 == _grad_checkpoint[_pointer - 1] + 1 )
            {
                _dequeue = 1;
                dynamic_size = _backward_exec[_sizepointer++];               
                BPS_LOG(DEBUG) << "enqueue operation of one stage is over." << "_sizepointer:" << _sizepointer << "mystack top is: " << _mystack.top();
                break;
                ///////////////////////////initialize dynamic size of this gradient stage.////////////////////////////
            }
          }
          // BPS_LOG(DEBUG) << "Position 4:"  << "_sq size is: "<< _sq.size();
  
          continue;
        }            
        // Size = Bandwidth * exectime , size decreased by the pop operation of mystack.
       // BPS_LOG(DEBUG) << "Task: " <<  task-> priority << "I have meet zero: " << _meetzero << " and door is open: " << _dooropen;
        
        if(task -> priority == 0) {
          _meetzero = 1;
         BPS_LOG(DEBUG) << "Meet zero." << "my stack size: " << _mystack.size();
         }
        if(!_meetzero)
        {
            if(task -> priority !=  _mystack.top())continue; 
            if(dynamic_size > task -> len){
              dynamic_size -= task -> len;
              BPS_LOG(DEBUG) << "dequeue element: " << task -> tensor_name << "dynamic size now is: " << dynamic_size;
              _sq.erase(it);
              _mystack.pop();
              BPS_LOG(DEBUG) << "PUSH gradient before 0: " << tmp ;
            }
            else{   //nxet stage enstack could begin.
              _dequeue = 0;
              _pointer--;
              _stagestart = 1;
              BytePSGlobal::pushsize[_sizepointer] = _mystack.top() + 1;
              BPS_LOG(DEBUG) << "PUSH: No left size. Waiting for next gradient block.";
              break;  
            }      
        }
        else if(_mywindow_size < task -> len ) {//we cannot change the value of tensor_part if door is closed.
          BPS_LOG(DEBUG) << "PUSH gradient after 0: " << tmp << "  window size" << _mywindow_size << "  window contents: " << _mywindow.size() << "  PUSH window is closed.";
          break;
        }
        else {         
            if(!_mystack.empty() && task -> priority !=  _mystack.top())continue;
            // _dooropen--;
            int ins = task -> priority * -1;
            if(!_mywindow.empty() && ins - *(_mywindow.begin()) > _difference_bound && _mywindow_size > _utilization_size)
              break;
            _mywindow_size -= task -> len;
            _mywindow.insert(task -> priority * -1);
            BPS_LOG(DEBUG) << "_mywindow.insert" << (task -> priority * -1);
            _sq.erase(it);
            _mystack.pop();
            // dynamic_size -= task -> len;  // if meetzero, dynamic size is no meaning.
            BPS_LOG(DEBUG) << "PUSH gradient after 0: " << tmp << " my window size: " << _mywindow_size ;
            // BPS_LOG(DEBUG) << "The door has been closed.";
          }
        //  BPS_LOG(DEBUG) << "transferred tensor num: " << _tensor_num  << "  empty: " << _mystack.empty() << " size of myqueue: " << _mystack.size();

        task->ready_event = nullptr;
        // Add for profiling communication TRACEs
        recorderTs(task);
        return task;
    }

    if (_is_scheduled) 
    {
        _credits -= task->len;
    }
    _sq.erase(it);
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                   << " getTask: " << task->tensor_name << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();  
    task->ready_event = nullptr;
    // Add for profiling communication TRACEs
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
    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                   << " getTask(key): " << task->tensor_name
                   << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication TRACEs
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
  std::lock_guard<std::mutex> lock(_mutex);
  if (_is_scheduled) {
      _credits += task ->  len;
  }
  std::string name = task -> tensor_name;

  if(_qt == PUSH && name.find("gradient") != name.npos) 
  {
    if(_meetzero) {
        BPS_LOG(DEBUG) << "PUSH element over:" << task ->tensor_name << "  mywindow size:" << _mywindow_size << " TOP element is: " <<  *(_mywindow.begin());
        if(_mywindow.lower_bound(task -> priority * -1) == _mywindow.end())
          return;
        _mywindow.erase(_mywindow.lower_bound(task -> priority * -1));
        _mywindow_size += task -> len;
        // _pullwindow.insert(task -> priority * -1);
        if(_mywindow.size() > 0 )
          BPS_LOG(DEBUG) << "after erase: " << "  mywindow size:" << _mywindow_size << " TOP element is: " << *(_mywindow.begin());    
        if(_mystack.empty() && _meetzero && _mywindow.size() == 0)
        {
            BPS_LOG(DEBUG) << "Clear.";
            _dequeue = 0;
            _pointer = 12;
            _stagestart = 1;
            _meetzero = 0;
            _sizepointer = 0;
            _mywindow_size = 8000000;
            //  _pushsize = 0;
        } 
    }
  }
  // if(_qt == PULL && name.find("gradient") != name.npos){
  //     if(_meetzero){
  //       if(_pullwindow.lower_bound(task -> priority * -1) == _pullwindow.end())
  //           return;
  //       _pullwindow.erase(_pullwindow.lower_bound(task -> priority * -1));
  
  //     }
  //   }
  return;
  }

}  // namespace common
}  // namespace byteps
