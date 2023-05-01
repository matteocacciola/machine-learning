#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <string.h>
#include "function.h"

#define min(X,Y)        ((X) < (Y) ? (X) : (Y))
#define max(X,Y)        ((X) > (Y) ? (X) : (Y))

double CalculateFitness(double);
void computeFitness(double *, int, int, double *);
double minArray(double *, int, int *);
double maxArray(double *, int, int *);
double mean(double *, int);
double r8UniformAB(double, double, int *);
int i4UniformAB(int, int, int *);
void init(int, int, double *, double *, double *, int *);
void keepTheBest(int, int, double *, double *, double *, double *);
void selector(int, int, double *, double *, double *, double *, int *);
void crossover(int, int, double, double *, int *);
void mutate(int, int, double, double *, double *, double *, int *);
void elitist(int, int, double *, double *, double *, double *);

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
 * 
 * @param genotype
 * @param popSize
 * @param D
 * @param fitness
 */
void computeFitness(double *genotype, int popSize, int D, double *fitness) {
    int i, j;
    double temp[D];

    /* calculate fitness for each set of possible weights */
#pragma omp parallel for shared(fitness, popSize, D, genotype) private(i, j, temp)
    for (i = 0; i < popSize; i++) {
        /* take the i-th set of genes from archive */
        for (j = 0; j < D; j++)
            temp[j] = genotype[i * D + j];
        fitness[i] = function(temp, D);
    }
}

/**
 * Min of an array. The function also returns the index of the minimum value
 * as per the C style pass by reference into *index
 * 
 * @param array
 * @param N
 * @param index
 * @return 
 */
double minArray(double *array, int N, int *index) {
    int i;
    double minimum = array[0];
    double minimumLocal = minimum;
    int indx = 0; /** NEW AFTER EMAIL GIUSEPPE 2 nov */
    int indxLocal = indx; /** NEW AFTER EMAIL GIUSEPPE 2 nov */
    
    if (N != 1) {
#pragma omp parallel shared(minimum, indx, array, N, index) firstprivate(minimumLocal, indxLocal)
        {
#pragma omp for
            for (i = 1; i < N; i++) {
                double dummy = array[i];
                if (minimumLocal > dummy) {
                    indxLocal = i;
                    minimumLocal = dummy;
                }
            }
#pragma omp critical
            {
                if (minimum > minimumLocal) {
                    indx = indxLocal;
                    minimum = minimumLocal;
                }
            } // end critical
        } // end of parallel
    }
    
    *index = indx; /** NEW AFTER EMAIL GIUSEPPE 2 nov */
    return minimum;
}

/**
 * Max of an array. The function also returns the index of the maximum value
 * as per the C style pass by reference into *index
 * 
 * @param array
 * @param N
 * @param index
 * @return 
 */
double maxArray(double *array, int N, int *index) {
    int i;
    double maximum = array[0];
    double maximumLocal = maximum;
    int indx = 0; /** NEW AFTER EMAIL GIUSEPPE 2 nov */
    int indxLocal = indx; /** NEW AFTER EMAIL GIUSEPPE 2 nov */
    
    if (N != 1) {
#pragma omp parallel shared(maximum, indx, array, N, index) firstprivate(maximumLocal, indxLocal)
        {
#pragma omp for
            for (i = 1; i < N; i++) {
                double dummy = array[i];
                if (maximumLocal < dummy) {
                    indxLocal = i;
                    maximumLocal = dummy;
                }
            }
#pragma omp critical
            {
                if (maximum < maximumLocal) {
                    indx = indxLocal;
                    maximum = maximumLocal;
                }
            } // end critical

        } // end parallel
    }
    
    *index = indx; /** NEW AFTER EMAIL GIUSEPPE 2 nov */
    return maximum;
}

/**
 * 
 * @param array
 * @param N
 * @return 
 */
double mean(double *array, int N) {
    int i;
    if (N != 1) {
        double sum = 0;
#pragma omp parallel for shared(array, N) private(i) reduction(+:sum)
        for (i = 0; i < N; i++)
            sum += array[i];
        return (double) (sum / N);
    } else {
        return array[0];
    }
}

/**
 * Returns a scaled pseudorandom R8
 * 
 * @param a
 * @param b
 * @param seed
 * @return 
 */
double r8UniformAB(double a, double b, int *seed) {
    int i4Huge = 2147483647;
    int k;
    double value;

    if (*seed == 0) {
        printf("ERROR!!! r8UniformAB fatal error!\nInput value of SEED = 0\n");
        exit(-1);
    }

    k = *seed / 127773;

    *seed = 16807 * (*seed - k * 127773) - k * 2836;

    if (*seed < 0) {
        *seed = *seed + i4Huge;
    }
    
    value = (double) (*seed) * 4.656612875E-10;
    value = a + (b - a) * value;

    return value;
}

/**
 * Return a scaled pseudorandom I4 between A and B
 * 
 * @param a
 * @param b
 * @param seed
 * @return 
 */
int i4UniformAB(int a, int b, int *seed) {
    int c, k, value;
    const int i4Huge = 2147483647;
    double r;

    if (*seed == 0) {
        printf("ERROR!!! i4UniformAB fatal error!\nInput value of SEED = 0\n");
        exit(-1);
    }

    /* Guarantee A <= B */
    if (b < a) {
        c = a;
        a = b;
        b = c;
    }

    k = *seed / 127773;

    *seed = 16807 * (*seed - k * 127773) - k * 2836;

    if (*seed < 0) {
        *seed = *seed + i4Huge;
    }

    r = (double) (*seed) * 4.656612875E-10;

    /* Scale R to lie between A-0.5 and B+0.5 */
    r = (1.0 - r) * ((double) a - 0.5) + r * ((double) b + 0.5);
    /* Use rounding to convert R to an integer between A and B */
    value = round(r);
    /* Guarantee A <= VALUE <= B */
    value = min(max(value, a), b);

    return value;
}

/**
 * Initialize the system
 * 
 * @param D
 * @param popSize
 * @param genotype
 * @param lb
 * @param ub
 * @param seed
 */
void init(int D, int popSize, double *genotype, double *lb, double *ub, int *seed) {
    int i, j;
#pragma omp parallel for shared(genotype, popSize, D, lb, ub, seed) private(i, j)
    for (i = 0; i < popSize; i++) {
        for (j = 0; j < D; j++) {
            genotype[i * D + j] = r8UniformAB(lb[j], ub[j], seed);
        }
    }
}

/**
 * Keep track of the best member of the population
 * 
 * @param D
 * @param popSize
 * @param genotype
 * @param fitness
 * @param bestFitness
 * @param bestElement
 */
void keepTheBest(int D, int popSize, double *genotype, double *fitness, double *bestFitness, double *bestElement) {
    int i, indx;

    *bestFitness = minArray(fitness, popSize, &indx);

    /* Once the best member in the population is found, copy the genes */
#pragma omp parallel for shared(bestElement, D, indx, genotype) private(i)
    for (i = 0; i < D; i++) {
        bestElement[i] = genotype[indx * D + i];
    }
}

/**
 * This is the selection function
 * 
 * @param D
 * @param popSize
 * @param genotype
 * @param fitness
 * @param rFitness
 * @param cFitness
 * @param seed
 */
void selector(int D, int popSize, double *genotype, double *fitness, double *rFitness, double *cFitness, int *seed) {
    const double a = 0.0;
    const double b = 1.0;
    int i, j, k;
    double temp[popSize * D];
    
#pragma omp parallel for shared(popSize, D, temp) private(i, j)
    for (i = 0; i < popSize; i++) {
        for (j = 0; j < D; j++) {
            temp[i * D + j] = 0;
        }
    }

    /* Find the total fitness of the population */
    double sum = 0.0;
#pragma omp parallel for shared(popSize, fitness) private(i) reduction(+:sum)
    for (i = 0; i < popSize; i++) {
        sum += fitness[i];
    }

    /* Calculate the relative fitness of each member */
#pragma omp parallel for shared(rFitness, fitness, popSize) private(i)
    for (i = 0; i < popSize; i++) {
        rFitness[i] = fitness[i] / sum;
    }

    /* Calculate the cumulative fitness */
    cFitness[0] = rFitness[0];
#pragma omp parallel for shared(cFitness, rFitness, popSize) private(i)
    for (i = 1; i < popSize; i++) {
        cFitness[i] = cFitness[i - 1] + rFitness[i];
    }

    /* Select survivors using cumulative fitness */
#pragma omp parallel for shared(popSize, cFitness, temp, genotype, D) private(i, j, k)
    for (i = 0; i < popSize; i++) {
        double p = r8UniformAB(a, b, seed);
        if (p < cFitness[0]) {
            for (k = 0; k < D; k++) {
                temp[i * D + k] = genotype[i * D + k];
            }
        } else {
            for (j = 0; j < (popSize - 1); j++) {
                if (cFitness[j] <= p && p < cFitness[j + 1]) {
                    for (k = 0; k < D; k++) {
                        temp[i * D + k] = genotype[(j + 1) * D + k];
                    }
                }
            }
        }
    }

    /*  Overwrite the old population with the new one */
    memmove(genotype, temp, popSize * D * sizeof (double));
}

/**
 * Select two parents for the single point crossover
 * 
 * @param D
 * @param popSize
 * @param pxOver
 * @param genotype
 * @param seed
 */
void crossover(int D, int popSize, double pxOver, double *genotype, int *seed) {
    const double a = 0.0;
    const double b = 1.0;
    int i, j, one;
    int first = 0;
    double x, point, t;

#pragma omp parallel for shared(genotype, one, pxOver, popSize, D, seed, first) private(i, j, x, point, t)
    for (i = 0; i < popSize; i++) {
        x = r8UniformAB(a, b, seed);

        if (x < pxOver) {
            first++;

            if (first % 2 == 0) {
                /* Select the crossover point */
                point = i4UniformAB(0, D - 1, seed);
                /* Swap genes in positions 0 through point-1 */
                for (j = 0; j < point; j++) {
                    t = genotype[one * D + j];
                    genotype[one * D + j] = genotype[i * D + j];
                    genotype[i * D + j] = t;
                }
            } else {
                one = i;
            }
        }
    }
}

/**
 * Perform a random uniform mutation
 * 
 * @param D
 * @param popSize
 * @param pMutation
 * @param genotype
 * @param lb
 * @param ub
 * @param seed
 */
void mutate(int D, int popSize, double pMutation, double *genotype, double *lb, double *ub, int *seed) {
    const double a = 0.0;
    const double b = 1.0;
    int i, j;
    double x;

#pragma omp parallel shared(genotype, popSize, D, seed, pMutation, lb, ub) private(i, j, x)
    for (i = 0; i < popSize; i++) {
        for (j = 0; j < D; j++) {
            x = r8UniformAB(a, b, seed);
            if (x < pMutation) {
                genotype[i * D + j] = r8UniformAB(lb[j], ub[j], seed);
            }
        }
    }
}

/**
 * Stores the best member of the previous generation
 * 
 * @param D
 * @param popSize
 * @param genotype
 * @param fitness
 * @param bestFitness
 * @param bestElement
 */
void elitist(int D, int popSize, double *genotype, double *fitness, double *bestFitness, double *bestElement) {
    int i, bestIndx, worstIndx;
    double best, worst;

    best = maxArray(fitness, popSize, &bestIndx);
    worst = minArray(fitness, popSize, &worstIndx);

    /*
     * If the best individual from the new population is better than the best
     * individual from the previous population, then copy the best from the new
     * population; else replace the worst individual from the current
     * population with the best one from the previous generation
     */
    if (*bestFitness <= best) {
        for (i = 0; i < D; i++) {
            bestElement[i] = genotype[bestIndx * D + i];
        }
        *bestFitness = best;
    } else {
#pragma omp parallel shared(genotype, D, worstIndx, bestElement) private(i)
        for (i = 0; i < D; i++) {
            genotype[worstIndx * D + i] = bestElement[i];
        }
        fitness[worstIndx] = *bestFitness;
    }
}

/**
 * Main program of PSO (Particle Swarm Optimization)
 * 
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char **argv) {
    printf("\t\t ---------------------------------------------\n");
    printf("\t\t|         WELCOME TO THE PROGRAM FOR          |\n");
    printf("\t\t|       GENETIC ALGORITHM OPTIMIZATION        |\n");
    printf("\t\t ---------------------------------------------\n");
    fflush(stdout);

    double sommaTempo, errorCriteria, tempoMedio, pxOver, pMutation, bestFitness,
            meanVal, actError, std, *genotype, *rFitness, *cFitness, *fitness,
            *bestElement, *worstfits, *bestfits, *meanfits, *minFits;
    int n, cycle, i, j, popSize, numOfDims, maxCycle, runtime;
    time_t start, stop;
    FILE *fd_ini, *fd_results_genes, *fd_results_global, *fd_results_mingenes;
    char *file_results_genes, *file_results_global, *file_results_mingenes;

    /* Allocate mem for filenames */
    if ((file_results_genes = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_genes)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_mingenes = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_mingenes)\n");
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
    fscanf(fd_ini, "D=%d\n", &numOfDims);

    if (numOfDims < 2) {
        printf("ERROR!!! The crossover modification will not be available, since it requires 2 <= numOfDims (main)\n");
    }

    fscanf(fd_ini, "popSize=%d\n", &popSize);
    fscanf(fd_ini, "pxOver=%lf\n", &pxOver);
    fscanf(fd_ini, "pMutation=%lf\n", &pMutation);

    double ub[numOfDims], lb[numOfDims];
    fscanf(fd_ini, "ub=%lf", &ub[0]);
    for (i = 1; i < numOfDims - 1; i++) {
        fscanf(fd_ini, ",%lf", &ub[i]);
    }
    fscanf(fd_ini, ",%lf\n", &ub[numOfDims - 1]);
    fscanf(fd_ini, "lb=%lf", &lb[0]);
    for (i = 1; i < numOfDims - 1; i++) {
        fscanf(fd_ini, ",%lf", &lb[i]);
    }
    fscanf(fd_ini, ",%lf\n", &lb[numOfDims - 1]);

    fscanf(fd_ini, "errorCriteria=%lf\n", &errorCriteria);
    fscanf(fd_ini, "runtime=%d\n", &runtime);
    fscanf(fd_ini, "maxCycle=%d\n", &maxCycle);
    fflush(fd_ini);
    fclose(fd_ini);

    /* allocateMem */
    printf("Allocating memory\n");
    fflush(stdout);
    if ((genotype = (double *) calloc(popSize * numOfDims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*genotype)\n");
        return (-1);
    }
    if ((rFitness = (double *) calloc(popSize, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*rFitness)\n");
        return (-1);
    }
    if ((cFitness = (double *) calloc(popSize, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*cFitness)\n");
        return (-1);
    }
    if ((fitness = (double *) calloc(popSize, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*fitness)\n");
        return (-1);
    }
    if ((bestElement = (double *) calloc(numOfDims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*bestElement)\n");
        return (-1);
    }

    if ((worstfits = (double *) calloc(maxCycle, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*worsts)\n");
        return (-1);
    }
    if ((bestfits = (double *) calloc(maxCycle, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*bests)\n");
        return (-1);
    }
    if ((meanfits = (double *) calloc(maxCycle, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*meanfits)\n");
        return (-1);
    }
    if ((minFits = (double *) calloc(runtime, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*GlobalMins)\n");
        return (-1);
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_genes, 40, "results_genes_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 40, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_mingenes, 40, "results_mingenes_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_genes = fopen(file_results_genes, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_mingenes = fopen(file_results_mingenes, "wb+");

    /* write headers into files */
    // results for particles
    fprintf(fd_results_genes, "#Runtime,#Cycle,#Genotype");
    for (i = 0; i < numOfDims; i++) {
        fprintf(fd_results_genes, ",Gene[%d]", i);
    }
    fprintf(fd_results_genes, "\n");
    // results for mingenes
    fprintf(fd_results_mingenes, "#Runtime,#Cycle");
    for (i = 0; i < numOfDims; i++) {
        fprintf(fd_results_mingenes, ",GeneBest[%d]", i);
    }
    fprintf(fd_results_mingenes, ",BestFit,Error\n");

    int seed = 123456789;

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    meanVal = 0;
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);
        fflush(stdout);

        /* initialize genotype */
        printf("\tInitializing genotype\n");
        fflush(stdout);
        init(numOfDims, popSize, genotype, lb, ub, &seed);

        computeFitness(genotype, popSize, numOfDims, fitness);
        keepTheBest(numOfDims, popSize, genotype, fitness, &bestFitness, bestElement);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            selector(numOfDims, popSize, genotype, fitness, rFitness, cFitness, &seed);
            crossover(numOfDims, popSize, pxOver, genotype, &seed);
            mutate(numOfDims, popSize, pMutation, genotype, lb, ub, &seed);

            computeFitness(genotype, popSize, numOfDims, fitness);
            elitist(numOfDims, popSize, genotype, fitness, &bestFitness, bestElement);

            int dummy;
            worstfits[cycle - 1] = minArray(fitness, popSize, &dummy);
            bestfits[cycle - 1] = bestFitness;
            meanfits[cycle - 1] = mean(fitness, popSize);

            actError = CalculateFitness(bestFitness);

            /* saving results into files */
            // particles
            for (i = 0; i < popSize; i++) {
                fprintf(fd_results_genes, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < numOfDims; j++) {
                    fprintf(fd_results_genes, ",%4.3f", genotype[i * numOfDims + j]);
                }
                fprintf(fd_results_genes, "\n");
            }
            // mingenes
            fprintf(fd_results_mingenes, "%d,%d", n + 1, cycle);
            for (i = 0; i < numOfDims; i++) {
                fprintf(fd_results_mingenes, ",%4.3f", bestElement[i]);
            }
            fprintf(fd_results_mingenes, ",%4.3f,%4.3f\n", bestFitness, actError);

            cycle++;
        } while ((cycle <= maxCycle) && (actError > errorCriteria));

        minFits[n] = bestFitness;
        meanVal += minFits[n];

        time(&stop);
        printf("\n\tElapsed time: %.3f s - Average GeneBestFit: %4.3f - Average GeneWorstFit: %4.3f - Average GeneMeanFit: %4.3f - MinFit: %4.3f\n",
                difftime(stop, start), mean(bestfits, cycle - 1), mean(worstfits, cycle - 1), mean(meanfits, cycle - 1), minFits[n]);
        fflush(stdout);
        
        // save info into global files
        fprintf(fd_results_global, "Runtime #%d, elapsed time %.3f s - Average GeneBestFit: %4.3f - Average GeneWorstFit: %4.3f - Average GeneMeanFit: %4.3f - MinFit: %4.3f\n",
                n + 1, difftime(stop, start), mean(bestfits, cycle - 1), mean(worstfits, cycle - 1), mean(meanfits, cycle - 1), minFits[n]);

        sommaTempo += difftime(stop, start);
    }

    meanVal = meanVal / runtime;
    tempoMedio = sommaTempo / runtime;
    std = 0;
    for (i = 0; i < runtime; i++) {
        std += (meanVal - minFits[i]) * (meanVal - minFits[i]);
    }
    std = sqrt(std / runtime);

    printf("Means of minFit of %d runs: %4.3f +/- %4.3f\nAverage optimization time: %.3f s\n", runtime, meanVal, std, tempoMedio);
    fflush(stdout);
    
    // save info into global files
    fprintf(fd_results_global, "Means of minFit of %d runs: %4.3f +/- %4.3f\nAverage optimization time: %.3f s\n", runtime, meanVal, std, tempoMedio);

    fflush(fd_results_genes);
    fclose(fd_results_genes);
    fflush(fd_results_mingenes);
    fclose(fd_results_mingenes);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(file_results_genes);
    free(file_results_mingenes);
    free(file_results_global);
    free(genotype);
    free(fitness);
    free(rFitness);
    free(cFitness);
    free(bestElement);
    free(worstfits);
    free(bestfits);
    free(meanfits);
    free(minFits);

    return (0);
}