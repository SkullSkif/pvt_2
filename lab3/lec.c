#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MSIZE 28000

void get_chunk(int a, int b, int commsize, int rank, int* lb, int* ub)
{
    /* OpenMP 4.0 spec (Sec. 2.7.1, default schedule for loops)
     * For a team of commsize processes and a sequence of n items, let ceil(n ?
     * commsize) be the integer q that satisfies n = commsize * q - r, with 0 <=
     * r < commsize. Assign q iterations to the first commsize - r procs, and q
     * - 1 iterations to the remaining r processes */
    int n = b - a + 1;
    int q = n / commsize;
    if (n % commsize)
        q++;
    int r = commsize * q - n;
    /* Compute chunk size for the process */
    int chunk = q;
    if (rank >= commsize - r)
        chunk = q - 1;
    *lb = a;        /* Determine start item for the process */
    if (rank > 0) { /* Count sum of previous chunks */
        if (rank <= commsize - r)
            *lb += q * rank;
        else
            *lb += q * (commsize - r) + (q - 1) * (rank - (commsize - r));
    }
    *ub = *lb + chunk - 1;
}

void* xmalloc(size_t size)
{
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr,
                "Error: Memory allocation failed for %zu bytes\n",
                size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/* dgemv: Compute matrix-vector product c[m] = a[m][n] * b[n] */
void dgemv(float* a, float* b, float* c, int m, int n)
{
    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int lb, ub;
    get_chunk(0, m - 1, commsize, rank, &lb, &ub);
    int nrows = ub - lb + 1;
    for (int i = 0; i < nrows; i++) {
        c[lb + i] = 0.0;
        for (int j = 0; j < n; j++)
            c[lb + i] += a[i * n + j] * b[j];
    }
    if (rank == 0) {
        int* displs = malloc(sizeof(*displs) * commsize);
        int* rcounts = malloc(sizeof(*rcounts) * commsize);
        for (int i = 0; i < commsize; i++) {
            int l, u;
            get_chunk(0, m - 1, commsize, i, &l, &u);
            rcounts[i] = u - l + 1;
            displs[i] = (i > 0) ? displs[i - 1] + rcounts[i - 1] : 0;
        }
        MPI_Gatherv(
                MPI_IN_PLACE,
                ub - lb + 1,
                MPI_FLOAT,
                c,
                rcounts,
                displs,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(
                &c[lb],
                ub - lb + 1,
                MPI_FLOAT,
                NULL,
                NULL,
                NULL,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv)
{
    int m = MSIZE;
    int n = MSIZE;
    int commsize, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t = MPI_Wtime();
    int lb, ub;
    get_chunk(
            0,
            m - 1,
            commsize,
            rank,
            &lb,
            &ub); // Декомпозиция матрицы на горизонтальные полосы
    int nrows = ub - lb + 1;
    float* a = xmalloc(sizeof(*a) * nrows * n);
    float* b = xmalloc(sizeof(*b) * n);
    float* c = xmalloc(sizeof(*c) * m);
    // Each process initialize their arrays
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = lb + i + 1;
    }
    for (int j = 0; j < n; j++)
        b[j] = j + 1;
    dgemv(a, b, c, m, n);
    t = MPI_Wtime() - t;
    if (rank == 0) {
        // Validation
        for (int i = 0; i < m; i++) {
            float r = (i + 1) * (n / 2.0 + pow(n, 2) / 2.0);
            if (fabs(c[i] - r) > 1E-6) {
                fprintf(stderr,
                        "Validation failed: elem %d = %f (real value %f)\n",
                        i,
                        c[i],
                        r);
                break;
            }
        }
        printf("DGEMV: matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n "
               "= %d)\n",
               m,
               n);
        printf("Memory used: %" PRIu64 " MiB\n",
               (uint64_t)(((float)m * n + m + n) * sizeof(float)) >> 20);
        double gflop = 2.0 * m * n * 1E-9;
        printf("Elapsed time (%d procs): %.6f sec.\n", commsize, t);
        printf("Performance: %.2f GFLOPS\n", gflop / t);
    }
    free(a);
    free(b);
    free(c);
    MPI_Finalize();
    return 0;
}