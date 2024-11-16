#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define f(x) ((log(1+x))/(x))

int main (int argc, char* argv[])
{
    MPI_Init (&argc, &argv);

    int rank, commsize;
    double eps = 1E-6; // Точность
    double n = 100; // Устанавливаем количество интервалов
    double a = 0.1; // Нижний предел
    double b = 1.0; // Верхний предел
    double h = (b - a) / n; // Вычисляем ширину интервала
    double summ_finale = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    double diff = 0.0;
    double timer_s = MPI_Wtime();
    for (int presc = 1; diff < eps; presc++)
    {
        double sum = 0.0;
        // Вычисляем границы для текущего процесса
        int left = presc * ((n / commsize) * rank);

        int right;
        if (rank != commsize - 1) {
            right = (presc * (n / commsize) * (rank + 1)) - 1;
        } else {
            right = (n * presc - 1);
        }
        // Суммируем значения функции в пределах текущего процесса
        for (int i = left; i <= right; i++)
        {
            sum += f(a + (h / 2) * i); // Вычисляем f(x) для середины интервала
        }
        // Объединяем суммы всех процессов
        MPI_Allreduce(&sum, &summ_finale, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        summ_finale = summ_finale * h;
        diff = summ_finale - diff;
    }

    double time_res = MPI_Wtime() - timer_s;
    if (rank == 0)
        printf("Result is: %f\n clculated in %f\n", summ_finale, time_res);
    MPI_Finalize();
    return 0;
}