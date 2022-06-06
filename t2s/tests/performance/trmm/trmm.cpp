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
#include "Halide.h"
#include "util.h"

// Constant parameters (inner loop bounds) of the design
#include "const-parameters.h"

using namespace Halide;

int main()
{
    // Dependences. We do not flatten i and k because k does not exist in P_Out.
    // We do not flatten i and j either, because we need remove j from A_serializer.
    #define P               kkk,      jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kkk_minus_1   kkk-1,    jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kk_minus_1    kkk+KKK-1,jjj,  iii,  jj, ii, kk-1,   k,  j,i
    #define P_k_minus_1     kkk+KKK-1,jjj,  iii,  jj, ii, kk+KK-1,k-1,j,i
    #define P_jjj_minus_1   kkk,      jjj-1,iii,  jj, ii, kk,     k,  j,i
    #define P_iii_minus_1   kkk,      jjj,  iii-1,jj, ii, kk,     k,  j,i
    #define P_Out                     jjj,  iii,  jj, ii,             j,i

    // Linearized addresses
    #define total_i         (iii + III * ii + III * II * i)
    #define total_j         (jjj + JJJ * jj + JJJ * JJ * j)
    #define total_k         (kkk + KKK * kk + KKK * KK * k)

    // For now, do not flatten i and k. Enable this feature in the next step, and
    // transfer only half of the matrix A.
    // #define i_k             ((((2 * K - i + 1) * i) / 2) + (k - i + 1))

    // Outer loop bounds, which are determined by input sizes
    #define I (A.dim(1).extent() / (III * II))
    #define J (B.dim(0).extent() / (JJJ * JJ))
    #define K (A.dim(0).extent() / (KKK * KK))

    // Type of the data to process in C and T2S
    #define CTYPE float
    #define TTYPE Float(32)

    // Inputs
    ImageParam A("A", TTYPE, 2), B("B", TTYPE, 2);

    // UREs
    Var kkk("kkk"), jjj("jjj"), iii("iii"), jj("jj"), ii("ii"), kk("kk"), k("k"), j("j"), i("i");
    URE X("X", TTYPE, {P}), Y("Y", TTYPE, {P}), Z("Z", TTYPE, {P}), Out("Out");
    X(P) = select(jjj == 0, A(total_k, total_i), X(P_jjj_minus_1));
    Y(P) = select(iii == 0, B(total_j, total_k), Y(P_iii_minus_1));
    Z(P) = select(kkk == 0 && kk == 0 && k == i, 0,
                select(kkk == 0, select(kk == 0, Z(P_k_minus_1), Z(P_kk_minus_1)), Z(P_kkk_minus_1)))
                + X(P) * Y(P);
    Out(P_Out) = select(kkk == KKK-1 && kk == KK-1 && k == K-1, Z(P));

    // Put all the UREs inside the same loop nest of X.
    X.merge_ures(Y, Z, Out);

    // Explicitly set the loop bounds
    X.set_bounds(jjj, 0, JJJ, iii, 0, III, kkk, 0, KKK)
     .set_bounds(jj,  0, JJ,  ii,  0, II,  kk,  0, KK)
     .set_bounds(j,   0, J,   i,   0, I,   k,   i, K - i);

    // Create a systolic array
    X.space_time_transform(jjj, iii);

    // Input path of matrix A
    Func ASerializer("ASerializer", Place::Host), ALoader("ALoader", Place::Device), AFeeder("AFeeder", Place::Device);
    X.isolate_producer_chain(A, ASerializer, ALoader, AFeeder);

    // For simplicity, let ASerializer send the full matrix A, even though ALoader still reads just the upper part of A.
    // That is, because the producer and consumer communicate through memory, instead of being directly connected by a channel,
    // the producer can write the memory in any way, and the consumer can read the memory in any way, as long as the producer
    // has finished all its writing to the memory before the consumer starts reading the memory.
    // TODO: enable sending only the upper triangle
    ASerializer.set_bounds(k, 0, K).remove(jjj, jj, j);

    // Input path of matrix B
    Func BSerializer("BSerializer", Place::Host), BLoader("BLoader", Place::Device), BFeeder("BFeeder", Place::Device);
    X.isolate_producer_chain(B, BSerializer, BLoader, BFeeder);
    BSerializer.set_bounds(k, 0, K).remove(iii, ii, i);

    Func deserializer("deserializer", Place::Host), unloader("unloader", Place::Device);
    Out.relay(Z, jjj);
    Out.isolate_consumer_chain(unloader, deserializer);
    unloader.vectorize(jjj);
    deserializer.vectorize(jjj);
    /*
     * T2X emits: X.isolate_producer_chain({A}, A_serializer, aLoader, aFeeder);
   2 T2X emits: aLoader.remove(jjj, jj);
   3 T2X emits: A_serializer.remove(jjj, jj, j);
   4 T2X emits: aFeeder.scatter(aLoader, iii);
   5 T2X emits: aFeeder.buffer(aLoader, k);
   6 T2X emits: aLoader.vectorize(kkk);
   7 T2X emits: aFeeder.vectorize(kkk);
   8 T2X emits: X.vectorize(kkk);
   9 T2X emits: aLoader.min_depth(256);
  10 T2X emits: aFeeder.min_depth(256);
  11 T2X emits: X.isolate_producer_chain({B}, B_serializer, bLoader, bFeeder);
  12 T2X emits: bLoader.remove(iii, ii);
  13 T2X emits: B_serializer.remove(iii, ii, i);
  14 T2X emits: bFeeder.scatter(bLoader, jjj);
  15 T2X emits: bFeeder.buffer(bLoader, k);
  16 T2X emits: bLoader.vectorize(kkk);
  17 T2X emits: bFeeder.vectorize(kkk);
  18 T2X emits: X.vectorize(kkk);
  19 T2X emits: bLoader.min_depth(256);
  20 T2X emits: bFeeder.min_depth(256);
  21 T2X emits: Out.relay(Z, jjj);
  22 T2X emits: Out.isolate_consumer_chain(unloader, deserializer);
  23 T2X emits: unloader.vectorize(jjj);
  24 T2X emits: deserializer.vectorize(jjj);
     *
     */

    // Compile the kernel to an FPGA bitstream, and expose a C interface for the host to invoke
    Target acc = get_host_target();
    acc.set_feature(Target::IntelFPGA);
    acc.set_feature(Target::EnableSynthesis);
    deserializer.compile_to_host("trmm-interface", { A, B }, "trmm", acc);

    printf("Success\n");
    return 0;
}
