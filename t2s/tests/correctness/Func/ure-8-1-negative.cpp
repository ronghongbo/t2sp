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
// A negative test case: Merge with undefined area

#include "util.h"

int main(void) {
    // Define the compute.
    ImageParam a(Int(32), 2, "a");
    ImageParam b(Int(32), 2, "b");
    Var i, j, k;
    Func A(Int(32), {i, j, k}), B(Int(32), {i, j, k}), C(Int(32), {i, j, k});

    // Define the compute.
    A(i, j, k) = select(j == 0, a(i, k), A(i, j - 1, k));
    C(i, j, k) = select(k == 0, A(i, j, k), C(i, j, k - 1) + A(i, j, k));

    try {
        A.merge_ures(B, C);
        B(i, j, k) = select(i == 0, b(k, j), B(i - 1, j, k));
    } catch (Halide::InternalError & e) {
        cout << e.what() << "Success!\n";
        exit(0);
    }
    cout << "Failed!\n";
    exit(1);
}
