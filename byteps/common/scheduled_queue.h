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
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <time.h>
#include <memory.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include "common.h"
#include "ready_table.h"


namespace byteps {
    namespace common {
        class BytePSScheduledQueue {
        public:
            BytePSScheduledQueue(QueueType type);

            QueueType getQueueType() { return _qt; }

            void addTask(std::shared_ptr <TensorTableEntry>);

            void recorderTs(std::shared_ptr <TensorTableEntry>);

            std::shared_ptr <TensorTableEntry> getTask();

            std::shared_ptr <TensorTableEntry> getTask(uint64_t key);

            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator findTask(int priority);

            uint32_t pendingSize();

            void reportFinish(std::shared_ptr <TensorTableEntry> task);

            unsigned long get_tcp_bytes();

        private:
            struct comparator {
                bool operator()(std::shared_ptr <TensorTableEntry> a, std::shared_ptr <TensorTableEntry> b) {
                    return (a->priority > b->priority);
                }
            };
            std::vector <std::shared_ptr<TensorTableEntry>> _sq;
            std::multiset <std::shared_ptr<TensorTableEntry>, comparator> _ms;
            std::vector <std::shared_ptr<TensorTableEntry>> _mysq;
            std::stack<int> _mystack;
            std::stack<int> _mystackpull;
            std::mutex _mutex;
            uint64_t _credits;
            uint64_t _init_credits;
            bool _is_scheduled;
            int _tensor_part[160] = {0};
            int _meetzero = 0;
            int _dooropen = 11;
            int _pulldoor = 0;
            int batchsize = 32;
            int _grad_checkpoint[13] = {-1, 9, 22, 35, 50, 62, 77, 90, 103, 117, 130, 143, 156};

            double _backward_exec[13] = {23.5, 23, 13, 15, 18.5, 26.5, 22, 32, 45,
                                      37, 29, 7.5, 0};
            int _exec_stage = 0;
            int _noleftsize = 0;
            int forward_dynamic_size;
            int _sizepointer = 0;
            int _stagepullnum = 0;
            int _dequeue = 0;
            int _pointer = 12;
            int _stagestart = 1;
            int dynamic_size;
            int _pushsize = 0;
            int _pullsize = 0;
            int expected_priority = _grad_checkpoint[_pointer];
            QueueType _qt;
            ReadyTable *_rt;
            //variable for monitor bandwidth 
            long last_time = 0;
            unsigned long last_tcp_size;
            double B = 125000 ;

        };
    }  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
