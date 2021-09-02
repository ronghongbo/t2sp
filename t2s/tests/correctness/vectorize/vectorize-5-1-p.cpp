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
// A positive test: producers in a chain vectorized with different sizes.

#include "util.h"

#define SIZE 32

int main(void) {
    // Define the compute.
    ImageParam a(Int(32), 1, "a");
    Func A(PLACE0);
    Var i;
    A(i) = a(i);
    A.set_bounds(i, 0, SIZE);

    Target target = get_host_target();
    target.set_feature(Target::IntelFPGA);

    // Generate input and run.
    Buffer<int> in = new_data<int, SIZE>(SEQUENTIAL); //or RANDOM
    a.set(in);
    Buffer<int> golden = A.realize(SIZE, target);

    // Isolate. Re-compile and run.
    Func A_feeder(PLACE1), A_loader(PLACE1);
    A.isolate_producer_chain(a, A_loader, A_feeder);
    A_loader.set_bounds(i, 0, SIZE);
    A_loader.vectorize(i, 4);
    A_feeder.set_bounds(i, 0, SIZE);
    A_feeder.vectorize(i, 2);
    A.vectorize(i, 16);
    remove(getenv("BITSTREAM"));
    Buffer<int> out = A.realize(SIZE, target);

    // Check correctness.
    check_equal<int>(golden, out);
    cout << "Success!\n";
    return 0;
}


