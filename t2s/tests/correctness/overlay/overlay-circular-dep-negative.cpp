/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the BSD-2-Clause Plus Patent License (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSDplusPatent
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: BSD-2-Clause-Patent
*******************************************************************************/
// A negative test case: circular dependency among tasks.
#include "util.h"

#define TYPE int
#define SIZE 16
#define N 4

int main(void) {
    // Declare 2 IPs on FPGA
    Func A0(Place::Device), A1(Place::Device);
    // Declare IPs' inputs
    ImageParam A(Int(32), 1);
    // Definitions of IPs
    Var iter;
    A0(iter) = A(iter) + 1;
    A1(iter) = A(iter) + 1;

    // Definitions of programming interface
    A0.command(0, INPUT_ONLY(), OUTPUT_ONLY(), INOUT(A));
    A1.command(1, INPUT_ONLY(), OUTPUT_ONLY(), INOUT(A));

    // Create overlay and compile
    Target target = get_host_target();
    target.set_feature(Target::IntelFPGA);

    auto overlay = Overlay(A0, A1).compile(target, "overlay");

    Func Task0(Place::Device), Task1(Place::Device);
    ImageParam a(Int(32), 1);
    Buffer<TYPE> inout = new_data<TYPE, SIZE>(SEQUENTIAL);
    Buffer<TYPE> result = new_data<TYPE, SIZE>(RANDOM);
    for (int i = 0; i < SIZE; i++) {
        result(i) = inout(i) + 2 * N;
    }
    a.set(inout);

    // Create tasks with actual inputs (buffers and values)
    Var i;
    Task0(i) = enqueue(overlay, 0, a);
    Task1(i) = enqueue(overlay, 1, a);
    Task0.merge_ures(Task1).set_bounds(i, 0, N);
    // There is a circular dependency
    Task0.depend(Task1, i);
    Task1.depend(Task0, i);

    inout.set_host_dirty();
    Task1.realize(target);
    inout.set_device_dirty();
    inout.copy_to_host();

    for (int j = 0; j < SIZE; j++) {
        printf("%d ", inout(j));
        printf("%d\n", result(j));
    }
    check_equal<TYPE>(result, inout);
    cout << "Success!\n";
}

void check(const Buffer<TYPE> &results, const Buffer<TYPE> &input) {
    bool pass = true;
    printf("*** Input:\n");
    for (int j = 0; j < SIZE; j++) {
        printf("%d ", input(j));
    }
    printf("\n");

    printf("*** Result:\n");
    for (int j = 0; j < SIZE; j++) {
        printf("%d ", results(j));
        bool correct = (results(j) == input(j) + 1);
        pass = pass && correct;
    }
    printf("\n");
    assert(pass);
}
