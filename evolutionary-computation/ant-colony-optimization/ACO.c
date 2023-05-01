#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <string.h>
#include <omp.h>
#include "function.h"

#define min(x,y)        ((x) < (y) ? (x) : (y))
#define max(x,y)        ((x) > (y) ? (x) : (y))
#define numEl(x)        (size_t)(sizeof x / sizeof *x)
#define sign(x)         (( x > 0 ) - ( x < 0 ))
#define PI 3.14159265358979323846

double CalculateFitness(double);
void sortOnCosts(int, int, double *, double *);
void init(int, int, double *, double *, double, double *, double *, double *, double *);
double randn(double, double);
void computeSigmas(int, int, double, double *, double *);
int selectPDF(int, double *);
void createArchive(int, int, int, double *, double *, double *, double *, double *, double *, double *);
void getOptimalPopulation(int, int, int, double *, double *, double *, double *);

/**
 * Fitness function
 * 
 * @param fun
 * @return 
 */
double CalculateFitness(double fun) {
    double result = (fun >= 0) ? 1 / (fun + 1) : 1 + fabs(fun);
    return result;
}

/**
 * Implementation of bubble sort algorithm to sort the passed population according
 * to the costs (fitnesses) of each solution.
 * 
 * @param n
 * @param ndims
 * @param positions
 * @param costs
 */
void sortOnCosts(int n, int ndims, double *positions, double *costs) {
    /** OLD SEQUENTIAL */
    /* int i, j, k;
    for (i = 0; i < (n - 1); i++) {
        for (j = 0; j < (n - 1 - i); j++) {
            if (costs[j] > costs[j + 1]) {
                double tempF, tempW;

                * swap fitness *
                tempF = costs[j];
                costs[j] = costs[j + 1];
                costs[j + 1] = tempF;

                * swap positions *
                for (k = 0; k < ndims; k++) {
                    tempW = positions[j * ndims + k];
                    positions[j * ndims + k] = positions[(j + 1) * ndims + k];
                    positions[(j + 1) * ndims + k] = tempW;
                }
            }
        }
    } */
    
    int i, j, k;
    double tempF, tempW;
    
    for (i = 0; i < n; i++) {
        int first = i % 2;
#pragma omp parallel for shared(costs, positions, n, ndims, first) private(j, k, tempF, tempW)
        for (j = first; j < n - 1; j += 2) {
            if (costs[j] > costs[j + 1]) {
                /* swap fitness */
                tempF = costs[j];
                costs[j] = costs[j + 1];
                costs[j + 1] = tempF;
                
                /* swap positions */
                for (k = 0; k < ndims; k++) {
                    tempW = positions[j * ndims + k];
                    positions[j * ndims + k] = positions[(j + 1) * ndims + k];
                    positions[(j + 1) * ndims + k] = tempW;
                }
            }
        }
    }
}

/**
 * Initialization of the system
 * 
 * @param nAnts
 * @param dim
 * @param a
 * @param b
 * @param positions
 * @param costs
 * @param w
 * @param p
 * @return 
 */
void init(int nAnts, int dim, double *a, double *b, double q, double *positions, double *costs, double *w, double *p) {
    double tempArr[dim], dummy, sigma, sumW, exponent, temp;
    int i, j;

    /* initialize populations: positions and costs */
    srand(time(NULL));
#pragma omp parallel for shared(positions, costs, nAnts, dim, a, b) private(i, j, dummy, tempArr)
    for (i = 0; i < nAnts; i++) {
        for (j = 0; j < dim; j++) {
            dummy = (double) rand() / ((double) RAND_MAX + 1.0) * (b[j] - a[j]) + a[j]; // from continuous uniform distribution
            positions[i * dim + j] = dummy;
            tempArr[j] = dummy;
        }
        costs[i] = function(tempArr, dim);
    }

    /* sorting initial population */
    sortOnCosts(nAnts, dim, positions, costs);

    /* calculating constant solutions weights */
    sigma = q * nAnts;
    sumW = 0;
#pragma omp parallel for shared(w, p, q, nAnts) private(i, exponent, temp) reduction(+:sumW)
    for (i = 0; i < nAnts; i++) {
        exponent = pow(i / sigma, 2) / 2;
        temp = (1 / (sqrt(2 * PI) * sigma)) * exp(-exponent);
        w[i] = temp;
        sumW += temp;
    }

#pragma omp parallel for shared(w, p, nAnts) private(i)
    for (i = 0; i < nAnts; i++) {
        p[i] = w[i] / sumW;
    }
}

/**
 * Function to generate a random number from a normal (Gaussian) distribution.
 * It exploits the Box-Muller (1958) transformation. It allows us to transform
 * uniformly distributed random variables, to a new set of random variables with
 * a Gaussian (or Normal) distribution.
 * 
 * Here, the polar form is implemented
 * 
 * @param miu
 * @param sigma
 * @return 
 */
double randn(double mu, double sigma) {
    srand(time(NULL));

    /*
    double x1, x2;
    x1 = ((double) rand() / ((double) (RAND_MAX) + 1.));
    x2 = ((double) rand() / ((double) (RAND_MAX) + 1.));

    double z;
    z = sqrt(-2 * log(x1)) * cos(PI * x2 * 2);
     */

    double x, w, y;
    do {
        x = 2.0 * ((double) rand() / ((double) (RAND_MAX) + 1.)) - 1.0;
        w = x * x;
    } while (w >= 1.0);

    w = sqrt((-2.0 * log(w)) / w);
    y = x * w;

    return (fabs(mu + sigma * y) > 1) ? sign(mu + sigma * y) : (mu + sigma * y);
}

/**
 * Compute standard deviations of distances (for each dimension of the problem)
 * among all possible couples of ants
 * 
 * @param N
 * @param D
 * @param zeta
 * @param sigmas
 * @param positions
 */
void computeSigmas(int N, int D, double zeta, double *sigmas, double *positions) {
    int i, j, k;
    double dists[D];
    
    memset(dists, 0, D * sizeof (double));

    /* distance among all possible ants */
#pragma omp parallel for shared(dists, N, D, positions) private(i, j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < D; k++) {
                dists[k] += fabs(positions[i * D + k] - positions[j * D + k]);
            }
        }
    }

    /* standard deviations */
#pragma omp parallel for shared(sigmas, N, D, zeta, dists) private(i, j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            sigmas[i * D + j] = (zeta * dists[j]) / (N - 1);
        }
    }
}

/**
 * select a Gaussian function that compose the Gaussian Kernel PDF
 * 
 * @param size
 * @param p
 * @return 
 */
int selectPDF(int size, double *p) {
    int i, l;
    double sum, r;

    srand(time(NULL));
    r = ((double) rand() / ((double) (RAND_MAX) + 1.));

    l = 0;
    sum = 0;
    i = 0;
    do {
        sum += p[i];
        i++;
    } while ((i < size) || (r > sum));

    /* assign the last valid i to l, after a RouletteWheelSelection procedure */
    l = i - 1;
    return l;
}

/**
 * Create a new archive and evaluate its members
 * 
 * @param S
 * @param N
 * @param D
 * @param lb
 * @param ub
 * @param archPos
 * @param archCosts
 * @param pos
 * @param sigmas
 * @param p
 */
void createArchive(int S, int N, int D, double *lb, double *ub, double *archPos, double *archCosts, double *pos, double *sigmas, double *p) {
    int i, j, l;
    double temp[D], dummy;

    /* archive construction */
    l = selectPDF(N, p); // select Gaussian Kernel
    // generate new positions around the formers with Gaussian Random Variable
#pragma omp parallel for shared(archPos, archCosts, S, D, pos, sigmas, lb, ub) private(i, j, dummy, temp)
    for (i = 0; i < S; i++) {
        for (j = 0; j < D; j++) {
            /* if generated dummy value is out of boundaries, it is shifted onto the boundaries */
            dummy = min(max(randn(pos[l * D + j], sigmas[l * D + j]), lb[j]), ub[j]);
            archPos[i * D + j] = dummy;
            temp[j] = dummy;
        }
        archCosts[i] = function(temp, D); // evaluate each archive
    }
}

/**
 * Get the optimal ants from merging the actual ones with the possible
 * archive's solutions
 * 
 * @param S
 * @param N
 * @param D
 * @param archPos
 * @param archCosts
 * @param popPos
 * @param popCosts
 */
void getOptimalPopulation(int S, int N, int D, double *archPos, double *archCosts, double *popPos, double *popCosts) {
    int i, j, offset;
    int nNewPop = (N + S);
    double tempPos[nNewPop * D], tempCosts[nNewPop];

    /* merge actual archive with actual population */
    // adding actual ants to the merging arrays
#pragma omp parallel for shared(tempPos, tempCosts, N, D, popPos, popCosts) private(i, j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            tempPos[i * D + j] = popPos[i * D + j];
        }
        tempCosts[i] = popCosts[i];
    }

    // adding actual archive's elements to the merging arrays
#pragma omp parallel for shared(tempPos, tempCosts, S, D, archPos, archCosts) private(i, j, offset)
    for (i = 0; i < S; i++) {
        offset = i + N;
        for (j = 0; j < D; j++) {
            tempPos[offset * D + j] = archPos[i * D + j];
        }
        tempCosts[offset] = archCosts[i];
    }

    /* sorting merged population array */
    sortOnCosts(nNewPop, D, tempPos, tempCosts);

    /* retain only N optimal values */
    memmove(popPos, tempPos, (N * D) * sizeof (double));
    memmove(popCosts, tempCosts, N * sizeof (double));
}

/**
 * Main program of ACO (Ant Colony Optimization)
 * 
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char **argv) {
    printf("\t\t ---------------------------------------------\n");
    printf("\t\t|         WELCOME TO THE PROGRAM FOR          |\n");
    printf("\t\t|           ANT COLONY OPTIMIZATION           |\n");
    printf("\t\t ---------------------------------------------\n");
    fflush(stdout);

    int n, cycle, i, j, archiveSize, nAnts, numofdims, runtime, maxCycle;
    double zeta, q, sommaTempo, tempoMedio, mean, std, errorCriteria, actError,
            bestCost, *popPositions, *popCosts, *archivePositions, *archiveCosts,
            *popWeights, *bestCosts, *probabilities, *sigmas;
    time_t start, stop;
    FILE *fd_ini, *fd_results_ants, *fd_results_global, *fd_results_minants;
    char *file_results_ants, *file_results_global, *file_results_minants;

    /* Allocate mem for filenames */
    if ((file_results_ants = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_ants)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_minants = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_minants)\n");
        return (-1);
    }

    /* read ini */
    if (argv[1] == NULL) {
        printf("ERROR!!! No ini file has been passed");
        return (-1);
    }
    printf("\nReading ini file (%s) with parameters\n", argv[1]);
    fflush(stdout);
    if ((fd_ini = fopen(argv[1], "r")) == NULL) {
        printf("ERROR!!! Open ini file error (main)\n");
        return (-1);
    }
    fscanf(fd_ini, "D=%d\n", &numofdims);
    fscanf(fd_ini, "archiveSize=%d\n", &archiveSize);
    fscanf(fd_ini, "nAnts=%d\n", &nAnts);
    fscanf(fd_ini, "q=%lf\n", &q);
    fscanf(fd_ini, "zeta=%lf\n", &zeta);

    double ub[numofdims], lb[numofdims];

    fscanf(fd_ini, "ub=%lf", &ub[0]);
    for (i = 1; i < numofdims - 1; i++) {
        fscanf(fd_ini, ",%lf", &ub[i]);
    }
    fscanf(fd_ini, ",%lf\n", &ub[numofdims - 1]);
    fscanf(fd_ini, "lb=%lf", &lb[0]);
    for (i = 1; i < numofdims - 1; i++) {
        fscanf(fd_ini, ",%lf", &lb[i]);
    }
    fscanf(fd_ini, ",%lf\n", &lb[numofdims - 1]);

    fscanf(fd_ini, "errorCriteria=%lf\n", &errorCriteria);
    fscanf(fd_ini, "runtime=%d\n", &runtime);
    fscanf(fd_ini, "maxCycle=%d\n", &maxCycle);
    fflush(fd_ini);
    fclose(fd_ini);

    /* allocateMem */
    printf("Allocating memory\n");
    fflush(stdout);
    if ((popPositions = (double *) calloc(nAnts * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*popPositions)\n");
        return (-1);
    }
    if ((popCosts = (double *) calloc(nAnts, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*popCosts)\n");
        return (-1);
    }
    if ((popWeights = (double *) calloc(nAnts, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*popWeights)\n");
        return (-1);
    }
    if ((bestCosts = (double *) calloc(maxCycle, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*bestCosts)\n");
        return (-1);
    }
    if ((probabilities = (double *) calloc(nAnts, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*probabilities)\n");
        return (-1);
    }
    if ((archivePositions = (double *) calloc(archiveSize * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*archivePositions)\n");
        return (-1);
    }
    if ((archiveCosts = (double *) calloc(archiveSize, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*archiveCosts)\n");
        return (-1);
    }
    if ((sigmas = (double *) calloc(nAnts * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*sigmas)\n");
        return (-1);
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_ants, 35, "results_ants_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 35, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_minants, 35, "results_minants_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_ants = fopen(file_results_ants, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_minants = fopen(file_results_minants, "wb+");

    /* write headers into files */
    // results for ants
    fprintf(fd_results_ants, "#Runtime,#Cycle,#Ant");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_ants, ",Dim[%d]", i);
    }
    fprintf(fd_results_ants, "\n");
    // results for minants
    fprintf(fd_results_minants, "#Runtime,#Cycle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_minants, ",Dim[%d]", i);
    }
    fprintf(fd_results_minants, ",BestCost,Error\n");

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    mean = 0;
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);
        fflush(stdout);

        /* initialize archive */
        printf("\tInitializing system\n");
        fflush(stdout);
        init(nAnts, numofdims, lb, ub, q, popPositions, popCosts, popWeights, probabilities);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            /* compute standard deviations of positions */
            computeSigmas(nAnts, numofdims, zeta, sigmas, popPositions);

            /* create and evaluate the archive of new possible solutions */
            createArchive(archiveSize, nAnts, numofdims, lb, ub, archivePositions, archiveCosts, popPositions, sigmas, probabilities);

            /* get the optimal population from the actual ones merged with the archive */
            getOptimalPopulation(archiveSize, nAnts, numofdims, archivePositions, archiveCosts, popPositions, popCosts);

            actError = CalculateFitness(popCosts[0]);

            /* saving results into files */
            // ants
            for (i = 0; i < nAnts; i++) {
                fprintf(fd_results_ants, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < numofdims; j++) {
                    fprintf(fd_results_ants, ",%4.3f", popPositions[i * numofdims + j]);
                }
                fprintf(fd_results_ants, "\n");
            }
            // minants
            fprintf(fd_results_minants, "%d,%d", n + 1, cycle);
            for (i = 0; i < numofdims; i++) {
                fprintf(fd_results_minants, ",%4.3f", popPositions[i]);
            }
            fprintf(fd_results_minants, ",%4.3f,%4.3f\n", popCosts[0], actError);

            cycle++;
        } while ((cycle <= maxCycle) && (actError > errorCriteria));

        bestCost = popCosts[0];

        bestCosts[n] = bestCost;
        mean += bestCost;

        time(&stop);
        printf("\n\tElapsed time: %.3f s - BestCost: %4.3f\n", difftime(stop, start), bestCost);
        fflush(stdout);

        // save info into global files
        fprintf(fd_results_global, "Runtime #%d, elapsed time %.3f s, BestCost: %4.3f\n",
                n + 1, difftime(stop, start), bestCost);

        sommaTempo += difftime(stop, start);
    }

    mean = mean / runtime;
    tempoMedio = sommaTempo / runtime;
    std = 0;
    for (i = 0; i < runtime; i++) {
        std += (mean - bestCosts[i]) * (mean - bestCosts[i]);
    }
    std = sqrt(std / runtime);

    printf("Means of BestCost of %d runs: %4.3f +/- %4.3f\nAverage optimization time: %.3f s\n", runtime, mean, std, tempoMedio);
    fflush(stdout);

    // save info into global files
    fprintf(fd_results_global, "Means of BestCost of %d runs +/- %4.3f: %4.3f\nAverage optimization time: %.3f s\n", runtime, mean, std, tempoMedio);

    fflush(fd_results_ants);
    fclose(fd_results_ants);
    fflush(fd_results_minants);
    fclose(fd_results_minants);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(file_results_ants);
    free(file_results_minants);
    free(file_results_global);
    free(popPositions);
    free(popCosts);
    free(popWeights);
    free(archivePositions);
    free(archiveCosts);
    free(bestCosts);
    free(probabilities);
    free(sigmas);

    return (0);
}