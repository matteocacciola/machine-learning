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
void init(int, int, double *, double *, double *, double *, double *, double *, double *);
void computeFitness(double *, int, int, double *);
void sortArchive(double *, double *, int, int);
double mean(double *, int);
void updParticles(double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, int, int, double, double, int, int, double, double);

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
 * Initialize particles
 * 
 * @param numofparticles
 * @param numofdims
 * @param Pos
 * @param lb
 * @param ub
 * @param V
 * @param Vmin
 * @param Vmax
 * @param fitness
 */
void init(int numofparticles, int numofdims, double *Pos, double *lb, double *ub, double *V, double *Vmin, double *Vmax, double *fitness) {
    int i, j;
    double lbJ, scale, VminJ, scaleV;
    
    srand(time(NULL));
#pragma omp parallel for shared(Pos, V, fitness, numofparticles, numofdims, Vmin, Vmax, lb, ub) private(i, j, lbJ, scale, VminJ, scaleV)
    for (i = 0; i < numofparticles; i++) {
        fitness[i] = INFINITY;
        for (j = 0; j < numofdims; j++) {
            lbJ = lb[j];
            scale = fabs(ub[j] - lbJ);
            VminJ = Vmin[j];
            scaleV = fabs(Vmax[j] - VminJ);
            // Pos[i * numofdims + j] = (double) (((rand() / (RAND_MAX + 1.))*2. - 1.)*(1 / sqrt((double) numofparticles)) * inizw);
            Pos[i * numofdims + j] = lbJ + (scale * ((double) rand() / ((double) (RAND_MAX) + 1.)));
            V[i * numofdims + j] = VminJ + (scaleV * ((double) rand() / ((double) (RAND_MAX) + 1.)));
        }
    }
}

/**
 * 
 * @param Pos
 * @param numofparticles
 * @param numofdims
 * @param fitness
 */
void computeFitness(double *Pos, int numofparticles, int numofdims, double *fitness) {
    int i, j;
    double temp[numofdims];

    /* calculate fitness for each set of possible weights */
#pragma omp parallel for shared(fitness, numofparticles, numofdims, Pos) private(i, j, temp)
    for (i = 0; i < numofparticles; i++) {
        /* take the i-th set of weights from archive */
        for (j = 0; j < numofdims; j++)
            temp[j] = Pos[i * numofdims + j];
        fitness[i] = function(temp, numofdims);
    }
}

/**
 * 
 * @param Pos
 * @param fitness
 * @param numofparticles
 * @param numofdims
 */
void sortArchive(double *Pos, double *fitness, int numofparticles, int numofdims) {
    int i, j, k;
    double tempF, tempW;
    
    for (i = 0; i < numofparticles; i++) {
        int first = i % 2;
#pragma omp parallel for shared(Pos, fitness, numofparticles, numofdims, first) private(j, k, tempF, tempW)
        for (j = first; j < numofparticles - 1; j += 2) {
            if (fitness[j] > fitness[j + 1]) {
                /* swap fitness */
                tempF = fitness[j];
                fitness[j] = fitness[j + 1];
                fitness[j + 1] = tempF;
                
                /* swap positions */
                for (k = 0; k < numofdims; k++) {
                    tempW = Pos[j * numofdims + k];
                    Pos[j * numofdims + k] = Pos[(j + 1) * numofdims + k];
                    Pos[(j + 1) * numofdims + k] = tempW;
                }
            }
        }
    }
}

/**
 * 
 * @param arr
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
 * 
 * @param fitness
 * @param pbestfits
 * @param pbests
 * @param gbest
 * @param Pos
 * @param V
 * @param Vmin
 * @param Vmax
 * @param lb
 * @param ub
 * @param numofparticles
 * @param numofdims
 * @param wmin
 * @param wmax
 * @param cycle
 * @param maxCycle
 * @param c1
 * @param c2
 */
void updParticles(double *fitness, double *pbestfits, double *pbests, double *gbest, double *Pos, double *V, double *Vmin, double *Vmax, double *lb, double *ub, int numofparticles, int numofdims, double wmin, double wmax, int cycle, int maxCycle, double c1, double c2) {
    int i, j;
    double temp;
    
    /* weight declining linearly */
    double w = wmax - (wmax - wmin) * (double) ((cycle + 1) / maxCycle);
#pragma omp parallel for shared(pbestfits, pbests, V, Pos, numofparticles, numofdims, fitness, w, c1, c2, Vmin, Vmax, lb, ub) private(i, j, temp)
    for (i = 0; i < numofparticles; i++) {
        pbestfits[i] = min(fitness[i], pbestfits[i]);
        for (j = 0; j < numofdims; j++) {
            temp = ((double) rand() / ((double) (RAND_MAX) + 1.));

            /* Update best positions before updating positions themselves */
            pbests[i * numofdims + j] = (fitness[i] < pbestfits[i]) ? Pos[i * numofdims + j] : pbests[i * numofdims + j];

            /* Update velocity and limit it by Vmax */
            V[i * numofdims + j] = min(max((w * V[i * numofdims + j] + temp * c1 * (pbests[i * numofdims + j] - Pos[i * numofdims + j]) + temp * c2 * (gbest[j] - Pos[i * numofdims + j])), Vmin[j]), Vmax[j]);

            /* Update position, and force them to be within the range [lb, ub] */
            Pos[i * numofdims + j] = min(max(Pos[i * numofdims + j] + V[i * numofdims + j], lb[j]), ub[j]);
        }
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
    printf("\t\t|         PARTICLE SWARM OPTIMIZATION         |\n");
    printf("\t\t ---------------------------------------------\n");
    fflush(stdout);

    double *Vmax, *Vmin, c1, c2, wmax, wmin, sommaTempo, errorCriteria, actError,
            tempoMedio, meanVal, std, *Pos, *fitness, *V, *pbests, *pbestfits,
            *gbest;
    int n, cycle, i, j, numofdims, numofparticles, maxCycle, runtime;
    time_t start, stop;
    FILE *fd_ini, *fd_results_particles, *fd_results_global, *fd_results_minparticles;
    char *file_results_particles, *file_results_global, *file_results_minparticles;

    /* Allocate mem for filenames */
    if ((file_results_particles = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_particles)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_minparticles = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_minparticles)\n");
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
    fscanf(fd_ini, "nParticles=%d\n", &numofparticles);
    fscanf(fd_ini, "c1=%lf\n", &c1);
    fscanf(fd_ini, "c2=%lf\n", &c2);
    fscanf(fd_ini, "wmax=%lf\n", &wmax);
    fscanf(fd_ini, "wmin=%lf\n", &wmin);

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
    if ((Pos = (double *) calloc(numofparticles * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*Pos)\n");
        return (-1);
    }
    if ((fitness = (double *) calloc(numofparticles, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*fitness)\n");
        return (-1);
    }
    if ((V = (double *) calloc(numofparticles * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*V)\n");
        return (-1);
    }
    if ((pbests = (double *) calloc(numofparticles * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*pbests)\n");
        return (-1);
    }
    if ((pbestfits = (double *) calloc(numofparticles, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*pbestfits)\n");
        return (-1);
    }
    if ((gbest = (double *) calloc(numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*gbest)\n");
        return (-1);
    }
    if ((Vmax = (double *) calloc(numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*Vmax)\n");
        return (-1);
    }
    if ((Vmin = (double *) calloc(numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*Vmin)\n");
        return (-1);
    }

    for (i = 0; i < numofdims; i++) {
        Vmax[i] = 0.2 * (ub[i] - lb[i]);
        Vmin[i] = -Vmax[i];
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_particles, 40, "results_particles_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 40, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_minparticles, 40, "results_minparticles_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_particles = fopen(file_results_particles, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_minparticles = fopen(file_results_minparticles, "wb+");

    /* write headers into files */
    // results for particles
    fprintf(fd_results_particles, "#Runtime,#Cycle,#Particle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_particles, ",PBest[%d]", i);
    }
    fprintf(fd_results_particles, "\n");
    // results for minparticles
    fprintf(fd_results_minparticles, "#Runtime,#Cycle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_minparticles, ",GBest[%d]", i);
    }
    fprintf(fd_results_minparticles, ",GBestFit,Error\n");

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    meanVal = 0;

    double worstfits[maxCycle], bestfits[maxCycle], meanfits[maxCycle], minFits[runtime];
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);
        fflush(stdout);

        /* initialize particle positions with random values for weights, within the range [-1 1] */
        printf("\tInitializing particles positions and velocities\n");
        fflush(stdout);
        init(numofparticles, numofdims, Pos, lb, ub, V, Vmin, Vmax, fitness);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            computeFitness(Pos, numofparticles, numofdims, fitness);
            sortArchive(Pos, fitness, numofparticles, numofdims);
            worstfits[cycle - 1] = fitness[numofparticles - 1];
            bestfits[cycle - 1] = fitness[0];
            meanfits[cycle - 1] = mean(fitness, numofparticles);

            for (i = 0; i < numofdims; i++) {
                gbest[i] = Pos[0 * numofdims + i];
            }

            actError = CalculateFitness(fitness[0]);

            /* saving results into files */
            // particles
            for (i = 0; i < numofparticles; i++) {
                fprintf(fd_results_particles, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < numofdims; j++) {
                    fprintf(fd_results_particles, ",%4.3f", Pos[i * numofdims + j]);
                }
                fprintf(fd_results_particles, "\n");
            }
            // minparticles
            fprintf(fd_results_minparticles, "%d,%d", n + 1, cycle);
            for (i = 0; i < numofdims; i++) {
                fprintf(fd_results_minparticles, ",%4.3f", gbest[i]);
            }
            fprintf(fd_results_minparticles, ",%4.3f,%4.3f\n", bestfits[cycle - 1], actError);

            updParticles(fitness, pbestfits, pbests, gbest, Pos, V, Vmin, Vmax, lb, ub, numofparticles, numofdims, wmin, wmax, cycle, maxCycle, c1, c2);

            cycle++;
        } while ((cycle <= maxCycle) && (actError > errorCriteria));

        minFits[n] = fitness[0];
        meanVal += minFits[n];

        time(&stop);
        printf("\n\tElapsed time: %.3f s - Average GBestFit: %4.3f - Average GWorstFit: %4.3f - Average GMeanFit: %4.3f - MinFit: %4.3f\n",
                difftime(stop, start), mean(bestfits, cycle - 1), mean(worstfits, cycle - 1), mean(meanfits, cycle - 1), minFits[n]);
        fflush(stdout);
        
        // save info into global files
        fprintf(fd_results_global, "Runtime #%d, elapsed time %.3f s - Average GBestFit: %4.3f - Average GWorstFit: %4.3f - Average GMeanFit: %4.3f - MinFit: %4.3f\n",
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

    fflush(fd_results_particles);
    fclose(fd_results_particles);
    fflush(fd_results_minparticles);
    fclose(fd_results_minparticles);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(file_results_particles);
    free(file_results_minparticles);
    free(file_results_global);
    free(Pos);
    free(fitness);
    free(V);
    free(pbests);
    free(pbestfits);
    free(gbest);
    free(Vmax);
    free(Vmin);

    return (0);
}