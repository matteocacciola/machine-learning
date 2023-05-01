#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <string.h>
#include <omp.h>
#include "function.h"

typedef int bool;

#define true 1
#define false 0
#define min(X,Y)        ((X) < (Y) ? (X) : (Y))
#define max(X,Y)        ((X) > (Y) ? (X) : (Y))

double sumArray(double *, int);
double mean(double *, int);
double CalculateFitness(double);
void init(int, int, double *, double *, double *, double *);
void computeFitness(double *, int, int, double *);
void getSortedChrome(int, double *, int, double *);
bool areEqualChromes(double *, double *, int);
void clearDups(int, int, double *, double *, double *);
void sortOnCosts(int, int, double *, double *);
double computeAverageCost(int, double *);
void elitism(int, double *, double *, double *, double *, int, int, bool);
void optimization(int, int, int, double, double, double, double, double *, double *, double *, double *);
double sumParamsOfChromes(double *, int, int, int);
void cauchyMutation(int, int, int, double, double *, double *, double *);

/**
 * sum of an array
 * 
 * @param array
 * @param N
 * @return 
 */
double sumArray(double *array, int N) {
    int i;
    double sum = 0;
#pragma omp parallel for shared(array, N) private(i) reduction(+:sum)
    for (i = 0; i < N; i++)
        sum += array[i];
    return sum;
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
 * Initialize the system
 * 
 * @param D
 * @param popSize
 * @param chrome
 * @param costs
 * @param lb
 * @param ub
 */
void init(int D, int popSize, double *chrome, double *costs, double *lb, double *ub) {
    int i, j;
    double scale, lbJ;
    
    srand(time(NULL));
#pragma omp parallel for shared(D, popSize, chrome, lb, ub) private(i, j, scale, lbJ)
    for (i = 0; i < popSize; i++) {
        for (j = 0; j < D; j++) {
            lbJ = lb[j];
            scale = fabs(ub[j] - lbJ);
            chrome[i * D + j] = lbJ + (scale * ((double) rand() / ((double) (RAND_MAX) + 1.)));
        }
    }

    // clearDups(popSize, D, chrome, lb, ub);
    computeFitness(chrome, popSize, D, costs); // it sorts from best to worst too
}

/**
 * Compute the fitness.
 * 
 * @param chrome
 * @param popSize
 * @param D
 * @param cost
 */
void computeFitness(double *chrome, int popSize, int D, double *cost) {
    int i, j;
    double temp[D];

    /* calculate fitness for each set of possible weights */
#pragma omp parallel for shared(cost, popSize, D, chrome) private(i, j, temp)
    for (i = 0; i < popSize; i++) {
        /* take the i-th set of weights from archive */
        for (j = 0; j < D; j++)
            temp[j] = chrome[i * D + j];
        cost[i] = function(temp, D);
    }

    /* Sort from best to worst */
    sortOnCosts(popSize, D, chrome, cost);
}

/**
 * Get the chromes of the index element of population, and sort it by using the
 * bubble sort algorithm.
 * 
 * @param D
 * @param chrome
 * @param index
 * @param chromeOfIndex
 */
void getSortedChrome(int D, double *chrome, int index, double *chromeOfIndex) {
    int i, j;
    double temp;
    
    memcpy(chromeOfIndex, chrome + index, D * sizeof (double));
    for (i = 0; i < D; i++) {
        int first = i % 2;
#pragma omp parallel for shared(chromeOfIndex, D, first) private(j, temp)
        for (j = first; j < D - 1; j += 2) {
            if (chromeOfIndex[j] > chromeOfIndex[j + 1]) {
                /* swap fitness */
                temp = chromeOfIndex[j];
                chromeOfIndex[j] = chromeOfIndex[j + 1];
                chromeOfIndex[j + 1] = temp;
            }
        }
    }
    
}

/**
 * Function that checks if the chromeA and chromeB are the same
 * 
 * @param chromeA
 * @param chromeB
 * @param D
 * @return 
 */
bool areEqualChromes(double *chromeA, double *chromeB, int D) {
    bool areEqual = true;
    int i;

    for (i = 0; i < D && areEqual == true; i++) {
        areEqual = (chromeA[i] == chromeB[i]) ? true : false;
    }

    return areEqual;
}

/**
 * Function to clear possible duplicates into chrome
 * 
 * @param popSize
 * @param D
 * @param chrome
 * @param lb
 * @param ub
 */
void clearDups(int popSize, int D, double *chrome, double *lb, double *ub) {
    int i, j, parnum;
    double chromeI[D], chromeJ[D];

#pragma omp parallel for shared(popSize, D, chrome, lb, ub, chromeI, chromeJ) private(i, j, parnum)
    for (i = 0; i < popSize; i++) {
        getSortedChrome(D, chrome, i, chromeI);
#pragma omp parallel for
        for (j = i + 1; j < popSize; j++) {
            getSortedChrome(D, chrome, j, chromeJ);
            if (areEqualChromes(chromeI, chromeJ, D)) {
                srand(time(NULL));
                parnum = floor(D * ((double) rand() / ((double) (RAND_MAX) + 1.)));
                chrome[j * D + parnum] = lb[parnum] + ((ub[parnum] - lb[parnum]) * ((double) rand() / ((double) (RAND_MAX) + 1.)));
            }
        }
    }
}

/**
 * Implementation of bubble sort algorithm to sort the passed population according
 * to the costs (fitnesses) of each chrome.
 * 
 * @param popSize
 * @param D
 * @param chrome
 * @param costs
 */
void sortOnCosts(int popSize, int D, double *chrome, double *costs) {
    int i, j, k;
    double tempF, tempW;
    
    for (i = 0; i < popSize; i++) {
        int first = i % 2;
#pragma omp parallel for shared(costs, chrome, popSize, D, first) private(j, k, tempF, tempW)
        for (j = first; j < popSize - 1; j += 2) {
            if (costs[j] > costs[j + 1]) {
                /* swap fitness */
                tempF = costs[j];
                costs[j] = costs[j + 1];
                costs[j + 1] = tempF;
                
                /* swap positions */
                for (k = 0; k < D; k++) {
                    tempW = chrome[j * D + k];
                    chrome[j * D + k] = chrome[(j + 1) * D + k];
                    chrome[(j + 1) * D + k] = tempW;
                }
            }
        }
    }
}

/**
 * Compute the average cost of legal chromes, i.e., chromes that do not have
 * INF values
 * 
 * @param popSize
 * @param costs
 * @return 
 */
double computeAverageCost(int popSize, double *costs) {
    int i;
    int nLegal = 0;
    double sum = 0;

#pragma omp parallel for shared(popSize, costs) private(i) reduction(+:sum, nLegal)
    for (i = 0; i < popSize; i++) {
        if (costs[i] < INFINITY) {
            nLegal += 1;
            sum += costs[i];
        }
    }

    return (sum / nLegal);
}

/**
 * Elitism strategy
 * 
 * @param howMany
 * @param chromeFrom
 * @param costsFrom
 * @param chromeTo
 * @param costsTo
 * @param popSize
 * @param D
 * @param reverseOrder
 */
void elitism(int howMany, double *chromeFrom, double *costsFrom, double *chromeTo, double *costsTo, int popSize, int D, bool reverseOrder) {
    int index = (reverseOrder == true) ? (popSize - howMany) : 0;
    memcpy(chromeTo, (chromeFrom) + (index * D), (howMany * D) * sizeof (double));
    memcpy(costsTo, (costsFrom) + index, howMany * sizeof (double));
}

/**
 * The earthworm optimization algorithm process
 * 
 * @param popSize
 * @param D
 * @param keep
 * @param alpha
 * @param beta
 * @param gamma
 * @param crossProb
 * @param chrome
 * @param costs
 * @param lb
 * @param ub
 */
void optimization(int popSize, int D, int keep, double alpha, double beta, double gamma, double crossProb, double *chrome, double *costs, double *lb, double *ub) {
    int i, j, mateA, selectIndex, r1Int;
    double inverseCosts[popSize], sumInverseCosts, oChrome[D], chromeTemp[D], randomCost, selectCost, r1Dbl, r2;

    beta = gamma * beta;

#pragma omp parallel for shared(popSize, costs, inverseCosts) private(i)
    for (i = 0; i < popSize; i++) {
        inverseCosts[i] = 1 / costs[i];
    }

    sumInverseCosts = sumArray(inverseCosts, popSize);

#pragma omp parallel for shared(popSize, D, lb, ub, alpha, beta, chrome, keep, sumInverseCosts, inverseCosts, crossProb) private(i, j, oChrome, chromeTemp, mateA, randomCost, selectCost, selectIndex, r1Dbl, r1Int, r2)
    for (i = 0; i < popSize; i++) {
        /* reproduction 1: the first way of reproducing */
        for (j = 0; j < D; j++) {
            oChrome[j] = ub[j] + lb[j] - alpha * chrome[i * D + j];
        }

        /* reproduction 2: the second way of reproducing */
        if (i >= keep + 1) {
            srand(time(NULL));
            mateA = round(0.5 + (popSize - 1) * ((double) rand() / ((double) (RAND_MAX) + 1.))) - 1;
            /* Select another parent to mate with the Earthworm r2 and create two children
             * roulette wheel selection
             * Make sure r2 is not selected as the second parent
             */
            randomCost = 0;
            selectCost = inverseCosts[0];
            while (selectCost >= randomCost) {
                srand(time(NULL));
                randomCost = sumInverseCosts * ((double) rand() / ((double) (RAND_MAX) + 1.));
            }
            selectIndex = 0;
            while (selectCost < randomCost) {
                selectIndex++;
                if (selectIndex >= popSize) {
                    break;
                }
                selectCost += inverseCosts[selectIndex];
            }
            /* Uniform crossover */
            for (j = 0; j < D; j++) {
                srand(time(NULL));
                r1Dbl = ((double) rand() / ((double) (RAND_MAX) + 1.));
                r2 = ((double) rand() / ((double) (RAND_MAX) + 1.));
                chromeTemp[j] = (crossProb > r1Dbl)
                        ? ((r2 > 0.5) ? chrome[mateA * D + j] : chrome[selectIndex * D + j])
                        : ((r2 > 0.5) ? chrome[selectIndex * D + j] : chrome[mateA * D + j]);
            }
        } else {
            srand(time(NULL));
            r1Int = round(0.5 + (popSize - 1) * ((double) rand() / ((double) (RAND_MAX) + 1.))) - 1;
            for (j = 0; j < D; j++) {
                chromeTemp[j] = chrome[r1Int * D + j];
            }
        }
        /* end of 2nd reproduction */

        /* Update chrome */
        for (j = 0; j < D; j++) {
            chrome[i * D + j] = beta * oChrome[j] + (1 - beta) * chromeTemp[j];
        }
    }
}

/**
 * Sum the parameters of all the chromes
 * 
 * @param chrome
 * @param parNum
 * @param popSize
 * @param D
 * @return 
 */
double sumParamsOfChromes(double *chrome, int parNum, int popSize, int D) {
    int i;
    double array[popSize];

#pragma omp parallel for shared(popSize, D, chrome, array, parNum) private(i)
    for (i = 0; i < popSize; i++) {
        array[i] = chrome[i * D + parNum];
    }

    return sumArray(array, popSize);
}

/**
 * Perform the Cauchy mutation (CM). It makes sure each individual is legal and
 * doesn't have duplicates
 * 
 * @param popSize
 * @param D
 * @param keep
 * @param mutateProb
 * @param chrome
 * @param lb
 * @param ub
 */
void cauchyMutation(int popSize, int D, int keep, double mutateProb, double *chrome, double *lb, double *ub) {
    int i, j;
    double bestChrome[D], cauchyW[D], sumJParam, dummy, r;

#pragma omp parallel for shared(D, bestChrome, cauchyW, chrome) private(i, dummy)
    for (i = 0; i < D; i++) {
        dummy = chrome[i];
        bestChrome[i] = dummy;
        cauchyW[i] = dummy;
    }

#pragma omp parallel for shared(keep, popSize, D, chrome, mutateProb, cauchyW, bestChrome, lb, ub) private(i, j, sumJParam, r)
    for (i = keep; i < popSize; i++) { // Don't allow the elites to be mutated
        for (j = 0; j < D; j++) {
            sumJParam = sumParamsOfChromes(chrome, j, popSize, D);

            srand(time(NULL));
            r = ((double) rand() / ((double) (RAND_MAX) + 1.));
            if (mutateProb > r) {
                cauchyW[j] = sumJParam / popSize;
            }

            chrome[i * D + j] = min(max(((cauchyW[j] + bestChrome[j]) / 2), lb[j]), ub[j]);
        }
    }
    
    clearDups(popSize, D, chrome, lb, ub);
}

/**
 * Main program of EWA (EarthWorm Algorithm)
 * 
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char **argv) {
    printf("\t\t ---------------------------------------------\n");
    printf("\t\t|         WELCOME TO THE PROGRAM FOR          |\n");
    printf("\t\t|            EARTHWORM ALGORITHM              |\n");
    printf("\t\t ---------------------------------------------\n");
    fflush(stdout);

    double pcross, pmutate, alpha, beta, gamma, errorCriteria, actError,
            sommaTempo, tempoMedio, meanVal, std, *chrome, *costs, *bestChrome;
    int n, cycle, i, j, numofdims, popsize, maxCycle, runtime, keep;
    time_t start, stop;
    FILE *fd_ini, *fd_results_worms, *fd_results_global, *fd_results_minworms;
    char *file_results_worms, *file_results_global, *file_results_minworms;

    /* Allocate mem for filenames */
    if ((file_results_worms = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_worms)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_minworms = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_minworms)\n");
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
    fscanf(fd_ini, "D=%d\n", &numofdims); // number of vriables in each population member
    fscanf(fd_ini, "popsize=%d\n", &popsize); // total population size
    fscanf(fd_ini, "pcross=%lf\n", &pcross); // crossover probability
    fscanf(fd_ini, "pmutate=%lf\n", &pmutate); // mutation probability
    fscanf(fd_ini, "keep=%d\n", &keep); // elitism parameter: how many of the best earthworm to keep from one generation to the next
    fscanf(fd_ini, "alpha=%lf\n", &alpha); // similarity factor
    fscanf(fd_ini, "beta=%lf\n", &beta); // the initial proportional factor
    fscanf(fd_ini, "gamma=%lf\n", &gamma); // a constant that is simliar to cooling factor of a cooling shudule in the simulated annealing.

    double lb[numofdims], ub[numofdims];
    
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
    if ((chrome = (double *) calloc(popsize * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*chrome)\n");
        return (-1);
    }
    if ((costs = (double *) calloc(popsize, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*cost)\n");
        return (-1);
    }
    if ((bestChrome = (double *) calloc(numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*bestChrome)\n");
        return (-1);
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_worms, 40, "results_worms_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 40, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_minworms, 40, "results_minworms_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_worms = fopen(file_results_worms, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_minworms = fopen(file_results_minworms, "wb+");

    /* write headers into files */
    // results for worms
    fprintf(fd_results_worms, "#Runtime,#Cycle,#Particle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_worms, ",EWABest[%d]", i);
    }
    fprintf(fd_results_worms, "\n");
    // results for minworms
    fprintf(fd_results_minworms, "#Runtime,#Cycle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_minworms, ",BestChrome[%d]", i);
    }
    fprintf(fd_results_minworms, ",BestChromeFit,Error\n");

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    meanVal = 0;

    double worstfits[maxCycle], bestfits[maxCycle], meanfits[maxCycle], minFits[runtime];
    double chromeKeep[keep * numofdims], costsKeep[keep];
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);
        fflush(stdout);

        /* initialize particle positions with random values for weights, within the range [-1 1] */
        printf("\tInitializing system\n");
        fflush(stdout);
        init(numofdims, popsize, chrome, costs, lb, ub);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            /* Elitism strategy: save the best earthworms in temporary arrays */
            elitism(keep, chrome, costs, chromeKeep, costsKeep, popsize, numofdims, false);
            /* The earthworm optimization algorithm process */
            optimization(popsize, numofdims, keep, alpha, beta, gamma, pcross, chrome, costs, lb, ub);
            /* Calculate costs, sorted from best to worst */
            computeFitness(chrome, popsize, numofdims, costs);
            /* Cauchy mutation (CM): make sure each individual is legal and doesn't have duplicates */
            cauchyMutation(popsize, numofdims, keep, pmutate, chrome, lb, ub);
            /* Calculate costs, sorted from best to worst */
            computeFitness(chrome, popsize, numofdims, costs);
            /* Elitism strategy: replace the worst with the previous generation's elites */
            elitism(keep, chrome, costs, chromeKeep, costsKeep, popsize, numofdims, true);

            worstfits[cycle - 1] = costs[popsize - 1];
            bestfits[cycle - 1] = costs[0];
            meanfits[cycle - 1] = computeAverageCost(popsize, costs);
            actError = CalculateFitness(costs[0]);

            for (i = 0; i < numofdims; i++) {
                bestChrome[i] = chrome[0 * numofdims + i];
            }

            /* saving results into files */
            // worms
            for (i = 0; i < popsize; i++) {
                fprintf(fd_results_worms, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < numofdims; j++) {
                    fprintf(fd_results_worms, ",%4.3f", chrome[i * numofdims + j]);
                }
                fprintf(fd_results_worms, "\n");
            }
            // minworms
            fprintf(fd_results_minworms, "%d,%d", n + 1, cycle);
            for (i = 0; i < numofdims; i++) {
                fprintf(fd_results_minworms, ",%4.3f", bestChrome[i]);
            }
            fprintf(fd_results_minworms, ",%4.3f,%4.3f\n", bestfits[cycle - 1], actError);

            cycle++;
        } while ((cycle <= maxCycle) && (actError > errorCriteria));

        minFits[n] = costs[0];
        meanVal += minFits[n];

        time(&stop);
        printf("\n\tElapsed time: %.3f s - Average BestChromeFit: %4.3f - Average WorstChromeFit: %4.3f - Average MeanFit: %4.3f - MinFit: %4.3f\n",
                difftime(stop, start), mean(bestfits, cycle - 1), mean(worstfits, cycle - 1), mean(meanfits, cycle - 1), minFits[n]);
        fflush(stdout);
        
        // save info into global files
        fprintf(fd_results_global, "Runtime #%d, elapsed time %.3f s - Average BestChromeFit: %4.3f - Average WorstChromeFit: %4.3f - Average MeanFit: %4.3f - MinFit: %4.3f\n",
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

    fflush(fd_results_worms);
    fclose(fd_results_worms);
    fflush(fd_results_minworms);
    fclose(fd_results_minworms);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(chrome);
    free(costs);
    free(bestChrome);
    free(file_results_worms);
    free(file_results_minworms);
    free(file_results_global);

    return (0);
}