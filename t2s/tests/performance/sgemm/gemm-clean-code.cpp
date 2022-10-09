#include "Halide.h"
#include "parameters.h"
using namespace Halide;
int main()
{
    #define P               kkk,      jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kkk_minus_1   kkk-1,    jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kk_minus_1    kkk+KKK-1,jjj,  iii,  jj, ii, kk-1,   k,  j,i
    #define P_k_minus_1     kkk+KKK-1,jjj,  iii,  jj, ii, kk+KK-1,k-1,j,i
    #define P_jjj_minus_1   kkk,      jjj-1,iii,  jj, ii, kk,     k,  j,i
    #define P_iii_minus_1   kkk,      jjj,  iii-1,jj, ii, kk,     k,  j,i
    #define P_Out                     jjj,  iii,  jj, ii,             j,i

    #define total_i         (iii + III * ii + III * II * i)
    #define total_j         (jjj + JJJ * jj + JJJ * JJ * j)
    #define total_k         (kkk + KKK * kk + KKK * KK * k)

    #define I (A.dim(1).extent() / (III * II))
    #define J (B.dim(0).extent() / (JJJ * JJ))
    #define K (A.dim(0).extent() / (KKK * KK))

    #define CTYPE float
    #define TTYPE Float(32)

    ImageParam A(TTYPE, 2), B(TTYPE, 2);

    Var kkk, jjj, iii, jj, ii, kk, k, j, i;
    URE X(TTYPE, {P}), Y(TTYPE, {P}), Z(TTYPE, {P}), Out;
    X(P) = select(jjj == 0, A(total_k, total_i), X(P_jjj_minus_1));
    Y(P) = select(iii == 0, B(total_j, total_k), Y(P_iii_minus_1));
    Z(P) = select(kkk == 0 && kk == 0 && k == 0, 0,
                select(kkk == 0, select(kk == 0, Z(P_k_minus_1), Z(P_kk_minus_1)), Z(P_kkk_minus_1)))
                + X(P) * Y(P);
    Out(P_Out) = select(kkk == KKK-1 && kk == KK-1 && k == K-1, Z(P));

    X.merge_ures(Y, Z, Out);
    X.set_bounds(jjj, 0, JJJ, iii, 0, III, kkk, 0, KKK)
     .set_bounds(jj,  0, JJ,  ii,  0, II,  kk,  0, KK)
     .set_bounds(j,   0, J,   i,   0, I,   k,   0, K);
    X.space_time_transform(jjj, iii);

    Stensor DA(DRAM), SA(SRAM), DB(DRAM), SB(SRAM), RC(REG), DC(DRAM), C;
    A >> DA.out(kkk) >> FIFO(256) >> SA.scope(k).out(kkk, iii)  >> FIFO(256);
    B >> DB.out(kkk) >> FIFO(256) >> SB.scope(k).out(kkk, jjj)  >> FIFO(256);
    Out >> RC.scope(iii).out(jjj) >> FIFO(256) >> DC >> C(total_j, total_i);

    C.compile_to_host("kernel-interface", { A, B }, "gemm", IntelFPGA);
    return 0;
}
