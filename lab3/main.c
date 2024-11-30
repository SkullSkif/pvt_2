#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>

void *xmalloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
// O(mn / p + n + m)
void get_chunk(int a, int b, int commsize, int rank, int *lb, int *ub)
{
    /* OpenMP 4.0 spec (Sec. 2.7.1, default schedule for loops)
     * For a team of commsize processes and a sequence of n items, let ceil(n ? commsize) be the integer q
     * that satisfies n = commsize * q - r, with 0 <= r < commsize.
     * Assign q iterations to the first commsize - r procs, and q - 1 iterations to the remaining r processes */
    int n = b - a + 1;
    int q = n / commsize;
    if (n % commsize)
        q++;
    int r = commsize * q - n;
    /* Compute chunk size for the process */
    int chunk = q;
    if (rank >= commsize - r)
        chunk = q - 1;
    *lb = a; /* Determine start item for the process */
    if (rank > 0)
    { /* Count sum of previous chunks */
        if (rank <= commsize - r)
            *lb += q * rank;
        else
            *lb += q * (commsize - r) + (q - 1) * (rank - (commsize - r));
    }
    *ub = *lb + chunk - 1;
}

void dgemv(float *a, float *b, float *c, int m, int n)
{
    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int lb, ub;
    get_chunk(0, m - 1, commsize, rank, &lb, &ub);
    int nrows = ub - lb + 1;

    for (int i = 0; i < nrows; i++)
    {
        c[lb + i] = 0.0;
        for (int j = 0; j < n; j++)
            c[lb + i] += a[i * n + j] * b[j];
    }

    int *displs = (int *)malloc(sizeof(*displs) * commsize);
    if (displs == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for displs\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int *rcounts = (int *)malloc(sizeof(*rcounts) * commsize);
    if (rcounts == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for rcounts\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < commsize; i++)
    {
        int l, u;
        get_chunk(0, m - 1, commsize, i, &l, &u);
        rcounts[i] = u - l + 1;                                   // Количество элементов от каждого процесса
        displs[i] = (i > 0) ? displs[i - 1] + rcounts[i - 1] : 0; // Смещений для каждой части
    }

    // Сбор всех частей вектора C
    MPI_Allgatherv(&c[lb], nrows, MPI_FLOAT, c, rcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);

    // Освобождаем память
    free(displs);
    free(rcounts);
}

int main(int argc, char **argv)
{
    int commsize, rank;
    int n = 45000;
    int m = 45000;
    double t;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        t = MPI_Wtime();
    int lb, ub;
    get_chunk(0, m - 1, commsize, rank, &lb, &ub); // Декомпозиция матрицы на горизонтальные полосы
    int nrows = ub - lb + 1;
    float *a = (float *)xmalloc(sizeof(*a) * nrows * n);
    float *b = (float *)xmalloc(sizeof(*b) * n);
    float *c = (float *)xmalloc(sizeof(*c) * m);
    // Each process initialize their arrays
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = lb + i + 1;
    }
    for (int j = 0; j < n; j++)
        b[j] = j + 1;

    dgemv(a, b, c, m, n);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        t = MPI_Wtime() - t;

    if (rank == 0)
    {
        printf("Count proc: %d\n", commsize);
        printf("DGEMV: matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
        printf("Proc memory used: %" PRIu64 " MiB\n", (uint64_t)((float)((m * n) / commsize + m + n) * sizeof(float)) >> 20);
        printf("Memory used: %" PRIu64 " MiB\n", (uint64_t)((float)(((m * n) / commsize + m + n) * commsize) * sizeof(float)) >> 20);
        printf("Elapsed time (%d procs): %lf sec.\n", commsize, t);
    }
    free(a);
    free(b);
    free(c);
    MPI_Finalize();
    return 0;
}