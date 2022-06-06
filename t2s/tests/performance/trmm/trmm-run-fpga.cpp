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
// The header file generated by trmm.cpp
#include "trmm-interface.h"

// Constant parameters (inner loop bounds) of the design
#include "const-parameters.h"

// Outer loop bounds for testing
#ifdef TINY // For verifying correctness only
    #define K           1//4
    #define J           1//4
    #define I           1//4
#else
    #define K           32
    #define J           32
    #define I           32
#endif

// Roofline utilities
#include "Roofline.h"

// The only header file needed for including T2S.
#include "HalideBuffer.h"

// For printing output
#include <stdio.h>
#include <iostream>

// For validation of results.
#include <assert.h>

using namespace std;

int main()
{
    const int TOTAL_I = III * II * I;
    const int TOTAL_J = JJJ * JJ * J;
    const int TOTAL_K = KKK * KK * K;

    assert(I == K);
    assert(TOTAL_I == TOTAL_K);

    Halide::Runtime::Buffer<float> a(TOTAL_K, TOTAL_I), b(TOTAL_J, TOTAL_K);
    for (size_t i = 0; i < TOTAL_I; i++) {
        for (size_t k = 0; k < TOTAL_K; k++) {
            a(k, i) = (k < i) ? 0 : k+i; //random();
        }
    }
    for (size_t k = 0; k < TOTAL_K; k++) {
        for (size_t j = 0; j < TOTAL_J; j++) {
            b(j, k) = j+k; //random();
        }
    }

#ifdef TINY
    // Validate the results
    Halide::Runtime::Buffer<float> golden(JJJ, III, JJ, II, J, I);
    for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
        for (int ii = 0; ii < II; ii++)
        for (int jj = 0; jj < JJ; jj++)
            for (int iii = 0; iii < III; iii++)
            for (int jjj = 0; jjj < JJJ; jjj++) {
                golden(jjj, iii, jj, ii, j, i) = 0.0f;
            }

    for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
    for (int k = i; k < K; k++)
        for (int kk = 0; kk < KK; kk++)
        for (int ii = 0; ii < II; ii++)
        for (int jj = 0; jj < JJ; jj++)
            for (int iii = 0; iii < III; iii++)
            for (int jjj = 0; jjj < JJJ; jjj++)
            for (int kkk = 0; kkk < KKK; kkk++) {
                size_t total_i = iii + III * ii + III * II * i;
                size_t total_j = jjj + JJJ * jj + JJJ * JJ * j;
                size_t total_k = kkk + KKK * kk + KKK * KK * k;
                printf("jjj=%i iii=%i jj=%i ii=%i j=%i i=%i: a=%f, b=%f\n",
                        jjj, iii, jj, ii, j, i, a(total_k, total_i), b(total_j, total_k)
                );
                golden(jjj, iii, jj, ii, j, i) += a(total_k, total_i) * b(total_j, total_k);
            }
#endif

    Halide::Runtime::Buffer<float> c(JJJ, III, JJ, II, J, I);
    trmm(a, b, c);

#ifdef TINY
    for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
        for (int ii = 0; ii < II; ii++)
        for (int jj = 0; jj < JJ; jj++)
            for (int iii = 0; iii < III; iii++)
            for (int jjj = 0; jjj < JJJ; jjj++) {
                printf("jjj=%i iii=%i jj=%i ii=%i j=%i i=%i: golden=%f, c=%f\n",
                        jjj, iii, jj, ii, j, i, golden(jjj, iii, jj, ii, j, i), c(jjj, iii, jj, ii, j, i)
                );
                assert(fabs(golden(jjj, iii, jj, ii, j, i) - c(jjj, iii, jj, ii, j, i))
                        < 0.005*fabs(golden(jjj, iii, jj, ii, j, i)));
            }

#endif

    printf("Success\n");
    return 0;
}
