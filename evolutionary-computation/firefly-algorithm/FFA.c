#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <string.h>
#include <omp.h>
#include "function.h"

#define min(X,Y)        ((X) < (Y) ? (X) : (Y))
#define max(X,Y)        ((X) > (Y) ? (X) : (Y))

double CalculateFitness(double);
void init(int, int, double *, double *, double *, double *);
double alphaNew(double, int);
void evaluate(int, int, double *, double *);
void sortOnCosts(int, double *, int *);
void replacePopulation(int, int, double *, double *, int *);
void moveFireflies(int, int, double, double, double, double *, double *, double *, double *, double *);

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
 * to the light intensity of each solution.
 * 
 * @param n
 * @param lightIntensity
 * @param index
 */
void sortOnCosts(int n, double *lightIntensity, int *index) {
    int i, j;
    double tempLI, tempI;
    
    // initialization of indexes
#pragma omp parallel for shared(index, n) private(i)
    for (i = 0; i < n; i++)
        index[i] = i;
    
    for (i = 0; i < n; i++) {
        int first = i % 2;
#pragma omp parallel for shared(index, lightIntensity, n, first) private(j, tempLI, tempI)
        for (j = first; j < n - 1; j += 2) {
            if (lightIntensity[j] > lightIntensity[j + 1]) {
                /* swap light intensity */
                tempLI = lightIntensity[j];
                lightIntensity[j] = lightIntensity[j + 1];
                lightIntensity[j + 1] = tempLI;
                
                /* swap indexes */
                tempI = index[j];
                index[j] = index[j + 1];
                index[j + 1] = tempI;
            }
        }
    }
}

/**
 * Initialization of the system
 * 
 * @param nFireflies
 * @param dim
 * @param a
 * @param b
 * @param ffa
 * @param lightIntensity
 * @return 
 */

void init(int nFireflies, int dim, double *a, double *b, double *ffa, double *lightIntensity) {
    int i, j;
    double dummy;
    
    memset(lightIntensity, 1.0, nFireflies * sizeof (double)); // initialize attractiveness

    srand(time(NULL));
#pragma omp parallel for shared(ffa, lightIntensity, nFireflies, dim, a, b) private(i, j, dummy)
    for (i = 0; i < nFireflies; i++) {
        for (j = 0; j < dim; j++) {
            dummy = (double) rand() / ((double) RAND_MAX + 1.0) * (b[j] - a[j]) + a[j]; // from continuous uniform distribution
            ffa[i * dim + j] = dummy;
        }
    }
}

/**
 * optionally recalculate the new alpha value
 * 
 * @param alpha
 * @param NGen
 * @return 
 */
double alphaNew(double alpha, int NGen) {
    double delta; // delta parameter
    delta = 1.0 - pow((pow(10.0, -4.0) / 0.9), 1.0 / (double) NGen);
    return (1 - delta) * alpha;
}

/**
 * Evaluate the population of fireflies
 * 
 * @param nFireflies
 * @param dim
 * @param ffa
 * @param lightIntensity
 */
void evaluate(int nFireflies, int dim, double *ffa, double *lightIntensity) {
    double tempArr[dim];
    int i, j;

#pragma omp parallel for shared(ffa, lightIntensity, nFireflies, dim) private(i, j, tempArr)
    for (i = 0; i < nFireflies; i++) {
        for (j = 0; j < dim; j++) {
            tempArr[j] = ffa[i * dim + j];
        }
        lightIntensity[i] = function(tempArr, dim); // obtain fitness of solution
    }
}

/**
 * Replace the population of fireflies
 * 
 * @param nFireflies
 * @param dim
 * @param ffa
 * @param ffaTmp
 * @param index
 */
void replacePopulation(int nFireflies, int dim, double *ffa, double *ffaTmp, int *index) {
    int i, j;

    // copy original population to temporary area
    memcpy(ffaTmp, ffa, nFireflies * dim * sizeof (double));

    // generational selection in sense of EA
#pragma omp parallel for shared(ffa, nFireflies, dim, index, ffaTmp) private(i, j)
    for (i = 0; i < nFireflies; i++) {
        for (j = 0; j < dim; j++) {
            ffa[i * dim + j] = ffaTmp[index[i] * dim + j];
        }
    }
}

/**
 * Move the population of fireflies
 * 
 * @param nFireflies
 * @param dim
 * @param alpha
 * @param betamin
 * @param gamma
 * @param a
 * @param b
 * @param ffa
 * @param ffaTmp
 * @param lightIntensity
 */
void moveFireflies(int nFireflies, int dim, double alpha, double betamin, double gamma, double *a, double *b, double *ffa, double *ffaTmp, double *lightIntensity) {
    int i, j, k;
    double scale, r, beta, tmpf, dummy;

    srand(time(NULL));
#pragma omp parallel for shared(nFireflies, dim, ffa, ffaTmp, alpha, betamin, gamma, a, b, lightIntensity) private(i, j, k, scale, r, beta, tmpf, dummy)
    for (i = 0; i < nFireflies; i++) {
        for (j = 0; j < nFireflies; j++) {
            r = 0.0;
            for (k = 0; k < dim; k++) {
                r += pow(ffa[i * dim + k] - ffa[j * dim + k], 2.0);
            }
            r = sqrt(r);
            if (lightIntensity[i] > lightIntensity[j]) { // brighter and more attractive
                beta = (1.0 - betamin) * exp(-gamma * pow(r, 2.0)) + betamin;
                for (k = 0; k < dim; k++) {
                    scale = fabs(b[k] - a[k]);
                    r = (double) rand() / ((double) RAND_MAX + 1.0);
                    tmpf = alpha * (r - 0.5) * scale;
                    dummy = min(max(ffa[i * dim + k] * (1.0 - beta) + ffaTmp[j * dim + k] * beta + tmpf, a[k]), b[k]);
                    ffa[i * dim + k] = dummy;
                }
            }
        }
    }
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
    printf("\t\t|              FIREFLY ALGORITHM              |\n");
    printf("\t\t ---------------------------------------------\n");

    int n, cycle, i, j, nFireflies, numofdims, runtime, maxCycle, *index;
    double alpha, betamin, gamma, sommaTempo, tempoMedio, mean, std, errorCriteria, actError,
            bestCost, *ffa, *ffaTmp, *lightIntensity, *bestCosts;
    time_t start, stop;
    FILE *fd_ini, *fd_results_fireflies, *fd_results_global, *fd_results_minfireflies;
    char *file_results_fireflies, *file_results_global, *file_results_minfireflies;

    /* Allocate mem for filenames */
    if ((file_results_fireflies = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_fireflies)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_minfireflies = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_minfireflies)\n");
        return (-1);
    }

    /* read ini */
    if (argv[1] == NULL) {
        printf("ERROR!!! No ini file has been passed");
        return (-1);
    }
    printf("\nReading ini file (%s) with parameters\n", argv[1]);
    if ((fd_ini = fopen(argv[1], "r")) == NULL) {
        printf("ERROR!!! Open ini file error (main)\n");
        return (-1);
    }
    fscanf(fd_ini, "D=%d\n", &numofdims);
    fscanf(fd_ini, "n=%d\n", &nFireflies);
    fscanf(fd_ini, "alpha=%lf\n", &alpha);
    fscanf(fd_ini, "betamin=%lf\n", &betamin);
    fscanf(fd_ini, "gamma=%lf\n", &gamma);

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
    if ((index = (int *) calloc(nFireflies, sizeof (int))) == NULL) {
        printf("ERROR!!! Not enough memory (*index)\n");
        return (-1);
    }
    if ((ffa = (double *) calloc(nFireflies * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*ffa)\n");
        return (-1);
    }
    if ((ffaTmp = (double *) calloc(nFireflies * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*ffaTmp)\n");
        return (-1);
    }
    if ((lightIntensity = (double *) calloc(nFireflies, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*lightIntensity)\n");
        return (-1);
    }
    if ((bestCosts = (double *) calloc(maxCycle, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*bestCosts)\n");
        return (-1);
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_fireflies, 45, "results_fireflies_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 45, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_minfireflies, 45, "results_minfireflies_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_fireflies = fopen(file_results_fireflies, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_minfireflies = fopen(file_results_minfireflies, "wb+");

    /* write headers into files */
    // results for fireflies
    fprintf(fd_results_fireflies, "#Runtime,#Cycle,#Firefly");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_fireflies, ",Dim[%d]", i);
    }
    fprintf(fd_results_fireflies, "\n");
    // results for minfireflies
    fprintf(fd_results_minfireflies, "#Runtime,#Cycle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_minfireflies, ",Dim[%d]", i);
    }
    fprintf(fd_results_minfireflies, ",BestCost,Error\n");

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    mean = 0;
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);

        /* initialize archive */
        printf("\tInitializing system\n");
        init(nFireflies, numofdims, lb, ub, ffa, lightIntensity);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            // this line of reducing alpha is optional
            alpha = alphaNew(alpha, maxCycle);

            // evaluate new solutions
            evaluate(nFireflies, numofdims, ffa, lightIntensity);

            // ranking fireflies by their light intensity
            sortOnCosts(nFireflies, lightIntensity, index);

            // replace old population
            replacePopulation(nFireflies, numofdims, ffa, ffaTmp, index);

            actError = CalculateFitness(lightIntensity[0]);

            /* saving results into files */
            // fireflies
            for (i = 0; i < nFireflies; i++) {
                fprintf(fd_results_fireflies, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < numofdims; j++) {
                    fprintf(fd_results_fireflies, ",%4.3f", ffa[i * numofdims + j]);
                }
                fprintf(fd_results_fireflies, "\n");
            }
            // minfireflies
            fprintf(fd_results_minfireflies, "%d,%d", n + 1, cycle);
            for (i = 0; i < numofdims; i++) {
                fprintf(fd_results_minfireflies, ",%4.3f", ffa[i]);
            }
            fprintf(fd_results_minfireflies, ",%4.3f,%4.3f\n", lightIntensity[0], actError);
            
            // move all fireflies to the best locations
            moveFireflies(nFireflies, numofdims, alpha, betamin, gamma, lb, ub, ffa, ffaTmp, lightIntensity);
            
            cycle++;
        } while ((cycle <= maxCycle) && (actError > errorCriteria));

        bestCost = lightIntensity[0];

        bestCosts[n] = bestCost;
        mean += bestCost;

        time(&stop);
        printf("\n\tElapsed time: %.3f s - BestCost: %4.3f\n", difftime(stop, start), bestCost);

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

    // save info into global files
    fprintf(fd_results_global, "Means of BestCost of %d runs +/- %4.3f: %4.3f\nAverage optimization time: %.3f s\n", runtime, mean, std, tempoMedio);

    fflush(fd_results_fireflies);
    fclose(fd_results_fireflies);
    fflush(fd_results_minfireflies);
    fclose(fd_results_minfireflies);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(file_results_fireflies);
    free(file_results_minfireflies);
    free(file_results_global);
    free(index);
    free(ffa);
    free(ffaTmp);
    free(lightIntensity);
    free(bestCosts);

    return (0);
}