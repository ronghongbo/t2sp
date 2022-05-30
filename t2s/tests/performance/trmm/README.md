# TRMM

In this test, we define TRMM as follows:

```
    C := alpha*A*B
```
where `alpha` is a non-zero scalar, `A` is a upper triangular matrix.

The design is the same as [GEMM](../gemm/README.md), except that `k` starts from `i`.

TODO: we should further tell the compiler that `i` and `k` loop can be flattened so that `A_serializer` sends only the tiles in the upper triangle sequentially. Therefore, in `A_loader`, we need translate the address of a tile referred to by `i` and `k` into a linearized address.

## [Test the design](../../../../README.md#Performance-tests)

# Thoughts on how to extend to the full and general TRMM in BLAS
The above simple TRMM can be extended to the full and general [TRMM in BLAS](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/trmm.html):

1. Add a BLAS interface on the host side. The interface implementation communicates data between the host and the device.
2. Overwrite `B`: the BLAS interface implementation receives `C` and writes into `B`.
3. Allow `alpha`: the BLAS interface implementation directly returns 0 matrix if `alpha` is 0.
4. Allow `A` to be in right side: the BLAS interface implementation transposes the compute if needed before inovking the device, and then transposes the results.
5. Allow `A` to be transposed: the BLAS interface implementation transposes `A` when specified.
6. Allow `A` to have unit diagonal: `A_loader` generates 1's for the diagonal. 
7. Allow leading dimensions of `A` and `B` to be specified: a linearized address of an `A` or `B` element should be calculated in term of the `lda` and `ldb`.
8. Allow `A` to be lower triangle: allow `k` to end at `i`.  