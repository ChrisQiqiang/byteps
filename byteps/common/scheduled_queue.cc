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

        unsigned long BytePSScheduledQueue::get_tcp_bytes(){
              unsigned long res;
          		std::ifstream fin("/proc/net/dev");                    
              while(!fin.eof())
              {
                std::string inbuf;
                int key_pos;
                std::getline(fin,inbuf,'\n');
                key_pos = inbuf.find("ens3",0);
                if(key_pos != std::string::npos && inbuf.find("pens3") == std::string::npos)
                {
                  std::string & str = inbuf.erase(0,key_pos);
                  unsigned long v;
                  float useage_net;
                  std::sscanf(str.c_str(),
                      "ens3:%lu %lu %lu %lu %lu %lu %lu %lu \
                        %lu %lu %lu %lu %lu %lu %lu %lu/n",
                        &v,&v,&v,&v,&v,&v,&v,&v,\
                        &res,&v,&v,&v,&v,&v,&v,&v);
                  fin.close();
                  break;
                }
              }	
              return res;            
        }

        BytePSScheduledQueue::BytePSScheduledQueue(QueueType type) {
            if ((type == REDUCE || type == PUSH) && BytePSGlobal::GetNccl()->IsSignalRoot()) {
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
                    _init_credits = _credits;
                    if(getenv("BANDWIDTH"))B = atoi(getenv("BANDWIDTH"));
                    if(getenv("Z_BATCH_SIZE")) batchsize = atoi(getenv("Z_BATCH_SIZE"));
                    for (int i = 0; i < 13; i++) 
                        _backward_exec[i] *= (double)batchsize / 32; 
                    // for (int i = 0; i < 13; i++) 
                    //     _backward_exec[i] *= B;
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
                    _sizepointer = 1;
                    break;
                default:
                    break;
            }
        }

        void BytePSScheduledQueue::addTask(std::shared_ptr <TensorTableEntry> entry) {
            std::lock_guard <std::mutex> lock(_mutex);
            if (_qt == PUSH && (entry->tensor_name).find("gradient") != (entry->tensor_name).npos) {
                _ms.insert(entry);
                _gradient_born = 1;
                _tensor_part[entry->priority * -1] = entry->total_partnum;
            } else {
                _sq.push_back(entry);
            }
            BPS_CHECK(entry->tensor_name != "");
            BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                           << " addTask: " << entry->tensor_name << " key: " << entry->key
                           << " rank: " << BytePSGlobal::GetLocalRank();
            return;
        }

        void BytePSScheduledQueue::recorderTs(std::shared_ptr <TensorTableEntry> task) {
            auto context = task->context;
            if (context->profile_flag) {
                auto now = std::chrono::system_clock::now();
                auto duration = now.time_since_epoch();
                auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

                auto &queue_list = task->queue_list;
                BPS_CHECK_GE(queue_list.size(), 1);
                auto this_op = queue_list[0];

                BPSCommTime *ret = new BPSCommTime;
                ret->start_t = (long long) (us.count());
                ret->key = task->key;
                ret->type = this_op;
                context->part_comm_time[task->key][this_op].push(ret);
            }
        }

        struct isTargetPriority {
            int Priority;

            isTargetPriority(int priority) : Priority(priority) {}

            bool operator()(std::shared_ptr <TensorTableEntry> x) {
                return x->priority == Priority;
            }
        };

        std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator BytePSScheduledQueue::findTask(int priority) {
            std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
            e->priority = priority;
            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator
            it = _ms.lower_bound(e);
            if (it == _ms.end()) {
                return it;
            } else if ((*it)->priority != priority) {
//                if (_tensor_part[priority * -1] == 0) {
//                    _tensor_part[priority * -1] = (*it)->total_partnum;
//                }
                return _ms.end();
            } else {
                BPS_CHECK_EQ((*it)->priority, priority);
                return it;
            }
        }




        std::shared_ptr <TensorTableEntry> BytePSScheduledQueue::getTask() {
            std::lock_guard <std::mutex> lock(_mutex);
            std::shared_ptr <TensorTableEntry> task;
            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator msit;
            if (_qt == PUSH && !_dequeue && _ms.size() > 0 && expected_priority >= 0) {
                while(true){
                      if(!_tensor_part[expected_priority]){
                          msit = findTask(expected_priority * -1);
                          if (msit == _ms.end()) 
                              return nullptr;
                          task = *msit;
                          _tensor_part[expected_priority] = task->total_partnum;
                      }
                      for (int x = 0; x < _tensor_part[expected_priority]; x++) 
                          _mystack.push(expected_priority * -1);
                      //BPS_LOG(INFO) << "has pushed element: " << expected_priority;
                      expected_priority--;
                      if (expected_priority == _grad_checkpoint[_pointer - 1]) {
                      //...............................................................................//
                        //initial variables for each stage.
                        long timenow;
                        unsigned long tcpsizenow;
                        struct timeval tmptime;
                        gettimeofday(&tmptime, NULL);
                        timenow = ((long)tmptime.tv_sec)*1000+(long)tmptime.tv_usec/1000;
                        //update B according to the last stage transfer information;
                        if(last_time != 0)
                            B = (get_tcp_bytes() - last_tcp_size) / (timenow - last_time);
                        dynamic_size = (int)(_backward_exec[_sizepointer++] * B);
                        _dequeue = 1;
                        BPS_LOG(INFO) << "dynamic size update: sizepointer" << _sizepointer << "  Bandwidth:" << B \
                                  <<" now dynamic size is:" << dynamic_size;
                        //BPS_LOG(INFO) << "last time is:" << last_time << "  time now:" << timenow;
                        //BPS_LOG(INFO) << "last tcp size:" << last_tcp_size << " tcp size now:" << get_tcp_bytes();
                        last_time = timenow;
                        last_tcp_size = get_tcp_bytes();
                        return nullptr;
                    }
                    if(expected_priority < 0)
                      return nullptr;
                }
                // BPS_LOG(INFO) << "DEAD LOOP!" ;
            }
            if (_qt == PUSH && _gradient_born) {
                if(_ms.size() == 0 && _mystack.empty() && _sizepointer < 12){
                    _dequeue = 0;
                    _pointer--;
                    BPS_LOG(INFO) << "has no element to push, wait for the next stage.";
                }
                else if(_ms.size() == 0){
                  BPS_LOG(INFO)  << " _ms size:" <<_ms.size() << "just return";
                  return nullptr;
                }
                msit = findTask(_mystack.top());
                if (msit == _ms.end()) {
                    return nullptr;
                }
                task = *msit;
                if (_sizepointer == 12) {
                    _meetzero = 1;
                }
                if (!_meetzero) {
                    if (dynamic_size > task->len) {
                        dynamic_size -= task->len;
                        _ms.erase(msit);
                        _mystack.pop();
                        if(_ms.size() < 10)
                          BPS_LOG(INFO)  << " _ms size:" <<_ms.size();
                    } else {
                        _dequeue = 0;
                        _pointer--;
                        BPS_LOG(INFO) << "PUSH for each stage is over. dequeue = 0" << " _ms size:" <<_ms.size();
                        //update backward_exec here according to real-time bandwidth monitor.
                        // BytePSGlobal::pushsize[_sizepointer] = _mystack.top() + 1;
                        return nullptr;
                    }
                }
                else if(_credits > task -> len){
                  _ms.erase(msit);
                  _mystack.pop();
                  _credits -= task->len;
                  BPS_LOG(INFO) << "credit size:" << _credits << " _ms size:" <<_ms.size();
                }
                else{
                  return nullptr;
                }
                task->ready_event = nullptr;
                recorderTs(task);
                return task;
            } else {
                for (auto it = _sq.begin(); it != _sq.end(); ++it) {
                    std::string tmp = (*it) -> tensor_name;
                    if ((*it)->ready_event) {
                        if (!(*it)->ready_event->Ready()) {
                            continue;
                        }
                    }
                    if (_is_scheduled && tmp.find("gradient") != tmp.npos) {
                        if ((*it)->len > _credits)
                            continue;
                    }
                    if (_rt) {
                        if (!_rt->IsKeyReady((*it)->key)) {
                            continue;
                        }
                        _rt->ClearReadyCount((*it)->key);
                    }
                    task = *it;
                    if (_is_scheduled && tmp.find("gradient") != tmp.npos) {
                        _credits -= task->len;
                    }
                    _sq.erase(it);
                    BPS_CHECK(task->tensor_name != "");
                    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                                   << " getTask: " << task->tensor_name << " key: " << task->key
                                   << " rank: " << BytePSGlobal::GetLocalRank();
                    task->ready_event = nullptr;
                    recorderTs(task);
                    return task;
                }
            }

            return nullptr;
        }

        std::shared_ptr <TensorTableEntry> BytePSScheduledQueue::getTask(uint64_t key) {
            BPS_CHECK(!_is_scheduled);
            std::lock_guard <std::mutex> lock(_mutex);
            std::shared_ptr <TensorTableEntry> task;
            for (auto it = _sq.begin(); it != _sq.end(); ++it) {
                if ((*it)->ready_event) {
                    BPS_CHECK((*it)->ready_event->Ready());
                }
                if ((*it)->key != (uint64_t) key) {
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
                recorderTs(task);
                return task;
            }
            return nullptr;
        }

        uint32_t BytePSScheduledQueue::pendingSize() {
            std::lock_guard <std::mutex> lock(_mutex);
            return _sq.size();
        }

        void BytePSScheduledQueue::reportFinish(std::shared_ptr <TensorTableEntry> task) {
            std::lock_guard <std::mutex> lock(_mutex);
            std::string tmp = task -> tensor_name;
            if ((_is_scheduled && _qt != PUSH) || (_qt == PUSH && tmp.find("gradient") != tmp.npos && _meetzero)) {
                _credits += task -> len;
            }
            if (_qt == PUSH && tmp.find("gradient") != tmp.npos && _mystack.empty() && _meetzero) {
                  _dequeue = 0;
                  _pointer = 12;
                  expected_priority = _grad_checkpoint[_pointer];
                  _meetzero = 0;
                  _sizepointer = 0;
                  _credits = _init_credits;
                  BPS_LOG(INFO) << "Clear." << "  credits: "<< _credits;
                  // _credits = BytePSGlobal::GetPartitionBound() * credit_in_partition;
                  // _dooropen = 11;
            }
            return;
        }

    }  // namespace common
}  // namespace byteps
