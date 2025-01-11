#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>

#define INF INT_MAX
#define V 3 // Количество вершин

void printSolution(int dist[][V]);

void floydWarshall(int graph[][V], int rank, int size) {
    int dist[V][V], i, j, k;

    // Инициализация расстояний
    for (i = 0; i < V; i++) {
        for (j = 0; j < V; j++) {
            dist[i][j] = graph[i][j];
        }
    }

    // Основной цикл алгоритма Флойда-Уоршелла
    for (k = 0; k < V; k++) {
        // Распространение информации о кратчайших путях между процессами
        for (i = rank; i < V; i += size) {
            for (j = 0; j < V; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
        // Обмен данными между процессами
        MPI_Allreduce(MPI_IN_PLACE, dist, V * V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printSolution(dist);
    }
}

void printSolution(int dist[][V]) {
    printf("Матрица кратчайших расстояний между всеми парами вершин:\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF) {
                printf("%7s", "INF");
            } else {
                printf("%7d", dist[i][j]);
            }
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Пример графа
        int graph[V][V] = {
            {0, 8, 5},
            {3, 0, INF},
            {INF, 2, 0}
        };

    // Каждый процесс получает граф
    int local_dist[V][V];
    MPI_Bcast(graph, V * V, MPI_INT, 0, MPI_COMM_WORLD);

    // Выполнение алгоритма Флойда-Уоршелла
    double t;
    t = MPI_Wtime();
    floydWarshall(graph, rank, size);
    t = MPI_Wtime() - t;
    if (rank == 0)
        printf("\nTime: %.6f\n", t);
    MPI_Finalize();
    return 0;
}