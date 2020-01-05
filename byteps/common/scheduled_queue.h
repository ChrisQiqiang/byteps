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

  enum model{resnet50, vgg16, transformer};
  std::vector<std::shared_ptr<TensorTableEntry>> _sq;
  std::vector<std::shared_ptr<TensorTableEntry>> _mysq;
  //add  myqueue to control addtask process.
  std::stack<int> _mystack;
  std::stack<int> _mystackpull;
  std::mutex _mutex;
  uint64_t _credits;
  bool _is_scheduled;
  int _tensor_part[160] = {0};//log every transferred tensor part
  int _meetzero = 0;
  int _exec_stage = 0;
  int _sizepointer = 0;
  int _stagepullnum = 0;
  int _dequeue = 0;
  int _stagestart = 1;
  int dynamic_size ;
  int _current_window_size;
  int _pointer;
  int batchsize = 32;
  std::multiset <int> _mywindow;
  QueueType _qt;
  ReadyTable *_rt;


  //parameters changes along by model change. Values for resnet50 are as follows. Default model is resnet50. 
  int _grad_checkpoint[13] = {-1, 9, 22, 35, 50, 62, 77, 90, 103, 117, 130, 143, 156}; //static for a specific model
  double _backward_exec[13] = {23.5, 23, 13, 15, 18.5, 26.5, 22, 32, 45, 37, 29, 0, 0}; // batchsize 32.
  int _mywindow_size = 8000000;
  int _utilization_size = 200000;
  int _difference_bound = 20;
  int _init_pointer = 11;
  int B = 125000;
  int _dooropen = 11;
  int iteration = 0;

};
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
