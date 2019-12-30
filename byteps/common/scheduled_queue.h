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

#ifndef BYTEPS_SCHEDULED_QUEUE_H
#define BYTEPS_SCHEDULED_QUEUE_H

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stack>
#include <set>
#include "common.h"
#include "ready_table.h"

namespace byteps {
namespace common {

class BytePSScheduledQueue {
 public:
  BytePSScheduledQueue(QueueType type);
  QueueType getQueueType() { return _qt; }
  void addTask(std::shared_ptr<TensorTableEntry>);
  void recorderTs(std::shared_ptr<TensorTableEntry>);
  std::shared_ptr<TensorTableEntry> getTask();
  std::shared_ptr<TensorTableEntry> getTask(uint64_t key);
  uint32_t pendingSize();
  void reportFinish(std::shared_ptr<TensorTableEntry> task);

 private:

  std::vector<std::shared_ptr<TensorTableEntry>> _sq;
  std::vector<std::shared_ptr<TensorTableEntry>> _mysq;
  //add  myqueue to control addtask process.
  std::stack<int> _mystack;
  std::stack<int> _mystackpull;
  std::mutex _mutex;
  uint64_t _credits;
  bool _is_scheduled;
  int _tensor_part[160] = {0};//log every transferred tensor part
  // int _tensor_num = 0; //log the number of transferred tensor.
  // int _vis[160] = {0};
  int _meetzero = 0;

  int _pulldoor = 0 ; 
  // int _grad_checkpoint[13] = {0,10,23,36,51,63,78,91,104,118,131,144,157};
  int _grad_checkpoint[13] = {-1,9,22,35,50,62,77,90,103,117,130,143,156};
  // int _backward_exec[13] = {2170000,2380000,1340000,1540000,2130000,2740000,2250000,3290000,4580000,3890000,2950000,0,0};
  int _backward_exec[13] = {5875000,5750000,3250000,3750000,4625000,6625000,5500000,8000000,11250000,9250000,7250000,0,0};
  // int _forward_exec[13] = {0,1350000,1400000,840000,900000,1275000,1620000,1335000,1900000,2700000,2200000,1750000,0};
  // int _forward_exec[13] = {0,2700000,2800000,1680000,1800000,2550000,3240000,2670000,3800000,5400000,4400000,3500000,0};
  int _exec_stage = 0;
  int _noleftsize = 0;
  int forward_dynamic_size;
  int _sizepointer = 0;
  int _stagepullnum = 0;

  int _dequeue = 0;
  int _pointer = 12;
  int _stagestart = 1;
  int dynamic_size ;
  int _pushsize = 0;
  int _pullsize = 0;
  // int xxx;


//forward parameter
  int _dooropen = 11; 
  std::multiset <int> _mywindow;
  int _mywindow_size = 8000000;
  int _utilization_size = 500000;
  int _difference_bound = 20;





  QueueType _qt;
  ReadyTable *_rt;
};
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
