#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    const double PI = 3.14159265358979323846;
    int n = 1000000;               // Общее количество точек для генерации.
    int in = 0;               // Переменная для хранения количества точек внутри области.
    int in_sum;                 // Переменная для хранения общего количества точек внутри области.
    double sum = 0.0;             // Переменная для накопления суммы.
    double n_sum;                 // Переменная для общей суммы, которая будет собрана с других процессов.
    int rank, commsize;   

    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(rank);
    double timer_s = MPI_Wtime();
    // Генерация случайных точек и подсчёт.
    for (int i = 0; i < n; i += commsize)
    {
        double x = (double)rand() / RAND_MAX;             // Генерируем x в диапазоне (0, 1).
        double y = (double)rand() / RAND_MAX * (1 - x);  // Генерируем y в диапазоне (0, 1 - x).

        if (y <= cos(x))                                  // Проверяем условие.
        {
            in++; // Увеличиваем счётчик точек внутри области.
            sum = sum + exp(pow((x + y), 2.0)); // Накопляем значение функции exp(x+y)^2.
        }
    }


    MPI_Reduce(&sum, &n_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Собираем сумму от всех процессов.
    MPI_Reduce(&in, &in_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); // Собираем количество точек внутри области.


    if (rank == 0)
    {    
        double v = PI * in_sum / n; // Вычисляем приближенную величину pi на основе накопленных точек.
        double res = v * n_sum / in_sum; // Рассчитываем конечный результат, нормализуя по количеству точек внутри области.
        double timer_res = MPI_Wtime() - timer_s;
        //printf("result is: %f\n n is: %d\n all dots inside: %d, time of work is: %f\n", res, n, in_sum, timer_res);
        printf("RESULT: %.10f \n n: %d \n TIME: %lf \n", res, n, timer_res);
    }

    MPI_Finalize();
    return 0;
}