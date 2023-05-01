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
void MemorizeBestSource(int, int, double *, double *, double *, double *);
void init(int, int, double *, double *, double *, double *, double *, double *, double *, double *);
void SendEmployedBees(int, int, double *, double *, double *, double *, double *, double *, double *);
void CalculateProbabilities(int, double *, double *);
void SendOnlookerBees(int, int, double *, double *, double *, double *, double *, double *, double *, double *);
void SendScoutBees(int, int, double *, double *, double *, double *, double *, double *, double *, int, double *, double *);

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
 * The best food source is memorized
 * 
 * @param FoodNumber
 * @param numofdims
 * @param f
 * @param Foods
 * @param GlobalParams
 * @param GlobalMin
 */
void MemorizeBestSource(int FoodNumber, int numofdims, double *f, double *Foods, double *GlobalParams, double *GlobalMin) {
    int i, j;
#pragma omp parallel for shared(GlobalMin, GlobalParams, FoodNumber, f, numofdims, Foods) private(i, j)
    for (i = 0; i < FoodNumber; i++) {
        if (f[i] < *GlobalMin) {
            *GlobalMin = f[i];
            for (j = 0; j < numofdims; j++)
                GlobalParams[j] = Foods[i * numofdims + j];
        }
    }
}

/**
 * Variables are initialized in the range [lb, ub]. If each parameter has different range,
 * use arrays lb[j], ub[j] instead of lb and ub
 * Counters of food sources are also initialized in this function
 * 
 * @param FoodNumber
 * @param numofdims
 * @param ub
 * @param lb
 * @param Foods
 * @param solution
 * @param f
 * @param fitness
 * @param GlobalMin
 * @param GlobalParams
 */
void init(int FoodNumber, int numofdims, double *lb, double *ub, double *Foods, double *solution, double *f, double *fitness, double *GlobalMin, double *GlobalParams) {
    int i, j;
    double temp1, temp2, lbJ, ubJ, scale;
    
    srand(time(NULL));
#pragma omp parallel for shared(Foods, solution, f, fitness, GlobalMin, GlobalParams, FoodNumber, numofdims, lb, ub) private(i, j, temp1, temp2, lbJ, ubJ, scale)
    for (i = 0; i < FoodNumber; i++) {
        for (j = 0; j < numofdims; j++) {
            lbJ = lb[j];
            ubJ = ub[j];
            scale = fabs(ubJ - lbJ);
            temp1 = ((double) rand() / ((double) (RAND_MAX) + 1.)) * scale + lbJ;
            Foods[i * numofdims + j] = temp1;
            solution[j] = temp1;
        }
        temp2 = function(solution, numofdims);
        f[i] = temp2;
        fitness[i] = CalculateFitness(temp2);
    }

    *GlobalMin = f[0];

#pragma omp parallel for shared(Foods, GlobalParams, numofdims) private(i)
    for (i = 0; i < numofdims; i++) {
        GlobalParams[i] = Foods[0 * numofdims + i];
    }
}

/**
 * 
 * @param FoodNumber
 * @param numofdims
 * @param lb
 * @param ub
 * @param Foods
 * @param solution
 * @param fitness
 * @param trial
 * @param f
 */
void SendEmployedBees(int FoodNumber, int numofdims, double *lb, double *ub, double *Foods, double *solution, double *fitness, double *trial, double *f) {
    int i, j, param2change, neighbour;
    double ObjValSol, FitnessSol;

    /* Employed Bee Phase */
    srand(time(NULL));
#pragma omp parallel for shared(solution, trial, Foods, f, fitness, numofdims, FoodNumber, lb, ub) private(i, j, param2change, neighbour, ObjValSol, FitnessSol)
    for (i = 0; i < FoodNumber; i++) {
        /* The parameter to be changed is determined randomly */
        param2change = (int) (((double) rand() / ((double) (RAND_MAX) + 1.)) * numofdims);

        /* A randomly chosen solution is used in producing a mutant solution of the solution i */
        neighbour = (int) (((double) rand() / ((double) (RAND_MAX) + 1.)) * FoodNumber);

        /* Randomly selected solution must be different from the solution i */
        while (neighbour == i) {
            neighbour = (int) (((double) rand() / ((double) (RAND_MAX) + 1.)) * FoodNumber);
        }

        for (j = 0; j < numofdims; j++)
            solution[j] = Foods[i * numofdims + j];
        /* v_{ij}=x_{ij}+\phi_{ij}*(x_{ij}-x_{kj}) */

        /* if generated parameter value is out of boundaries, it is shifted onto the boundaries */
        solution[param2change] = min(max(Foods[i * numofdims + param2change] + (Foods[i * numofdims + param2change] - Foods[neighbour * numofdims + param2change]) * (((double) rand() / ((double) (RAND_MAX) + 1.)) - 0.5) * 2, lb[param2change]), ub[param2change]);
        ObjValSol = function(solution, numofdims);
        FitnessSol = CalculateFitness(ObjValSol);

        /* a greedy selection is applied between the current solution i and its mutant */
        if (FitnessSol > fitness[i]) {
            /* If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i */
            trial[i] = 0;
            for (j = 0; j < numofdims; j++)
                Foods[i * numofdims + j] = solution[j];
            f[i] = ObjValSol;
            fitness[i] = FitnessSol;
        } else {
            /* if the solution i can not be improved, increase its trial counter */
            trial[i] = trial[i] + 1;
        }
    }
}

/**
 * A food source is chosen with the probability which is proportional to its
 * quality.
 * Different schemes can be used to calculate the probability values
 * For example prob(i)=fitness(i)/sum(fitness)
 * or in a way used in the method below prob(i)=a*fitness(i)/max(fitness)+b
 * 
 * probability values are calculated by using fitness values and normalized by
 * dividing maximum fitness value
 * 
 * @param FoodNumber
 * @param fitness
 * @param prob
 */
void CalculateProbabilities(int FoodNumber, double *fitness, double *prob) {
    int i;
    double sumfit = 0; /* double maxfit = -INFINITY; */
#pragma omp parallel for shared(FoodNumber, fitness) private(i) reduction(+:sumfit)
    for (i = 0; i < FoodNumber; i++) {
        sumfit += fitness[i]; /* maxfit = max(fitness[i], maxfit); */
    }
#pragma omp parallel for shared(prob, FoodNumber, fitness, sumfit) private(i)
    for (i = 0; i < FoodNumber; i++) {
        prob[i] = fitness[i] / sumfit; /* prob[i] = (0.9 * (fitness[i] / maxfit)) + 0.1; */
    }
}

/**
 * 
 * @param FoodNumber
 * @param numofdims
 * @param lb
 * @param ub
 * @param Foods
 * @param solution
 * @param fitness
 * @param trial
 * @param f
 * @param prob
 */
void SendOnlookerBees(int FoodNumber, int numofdims, double *lb, double *ub, double *Foods, double *solution, double *fitness, double *trial, double *f, double *prob) {
    int i, j, k, param2change, neighbour;
    double r, ObjValSol, FitnessSol;
    
    i = 0;
    k = 0;
    srand(time(NULL));
    /* onlooker Bee Phase */
    while (k < FoodNumber) {
        r = ((double) rand() / ((double) (RAND_MAX) + 1.));

        /* choose a food source depending on its probability to be chosen */
        if (r < prob[i]) {
            k++;

            /* The parameter to be changed is determined randomly */
            param2change = (int) (((double) rand() / ((double) (RAND_MAX) + 1.)) * numofdims);

            /* A randomly chosen solution is used in producing a mutant solution of the solution i */
            neighbour = (int) (((double) rand() / ((double) (RAND_MAX) + 1.)) * FoodNumber);

            /* Randomly selected solution must be different from the solution i */
            while (neighbour == i) {
                neighbour = (int) (((double) rand() / ((double) (RAND_MAX) + 1.)) * FoodNumber);
            }

            for (j = 0; j < numofdims; j++)
                solution[j] = Foods[i * numofdims + j];
            /* v_{ij}=x_{ij}+\phi_{ij}*(x_{ij}-x_{kj}) */

            /* if generated parameter value is out of boundaries, it is shifted onto the boundaries */
            solution[param2change] = min(max(Foods[i * numofdims + param2change] + (Foods[i * numofdims + param2change] - Foods[neighbour * numofdims + param2change]) * (((double) rand() / ((double) (RAND_MAX) + 1.)) - 0.5) * 2, lb[param2change]), ub[param2change]);
            ObjValSol = function(solution, numofdims);
            FitnessSol = CalculateFitness(ObjValSol);

            /* a greedy selection is applied between the current solution i and its mutant */
            if (FitnessSol > fitness[i]) {
                /*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i */
                trial[i] = 0;
                for (j = 0; j < numofdims; j++)
                    Foods[i * numofdims + j] = solution[j];
                f[i] = ObjValSol;
                fitness[i] = FitnessSol;
            } else {
                /* if the solution i can not be improved, increase its trial counter */
                trial[i] = trial[i] + 1;
            }
        }
        i++;

        if (i == FoodNumber) {
            i = 0;
        }
    }
}

/**
 * Determine the food sources whose trial counter exceeds the "limit" value. 
 * In Basic BCO, only one scout is allowed to occur in each cycle
 * 
 * @param FoodNumber
 * @param numofdims
 * @param lb
 * @param ub
 * @param Foods
 * @param solution
 * @param f
 * @param fitness
 * @param trial
 * @param limit
 */
void SendScoutBees(int FoodNumber, int numofdims, double *lb, double *ub, double *Foods, double *solution, double *f, double *fitness, double *trial, int limit, double *GlobalMin, double *GlobalParams) {
    int maxtrialindex, i;
    maxtrialindex = 0;
#pragma omp parallel for shared(maxtrialindex, FoodNumber, limit, numofdims, lb, ub, Foods, solution, f, fitness, trial, GlobalMin, GlobalParams) private(i)
    for (i = 1; i < FoodNumber; i++) {
        maxtrialindex = (trial[i] > trial[maxtrialindex]) ? i : maxtrialindex;
    }
    if (trial[maxtrialindex] >= limit) {
        trial[maxtrialindex] = 0;
        init(maxtrialindex, numofdims, lb, ub, Foods, solution, f, fitness, GlobalMin, GlobalParams);
    }
}

/**
 * Main program of BCO (Bee Colony Optimization)
 * 
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char **argv) {
    printf("\t\t ---------------------------------------------\n");
    printf("\t\t|         WELCOME TO THE PROGRAM FOR          |\n");
    printf("\t\t|          BEE COLONY OPTIMIZATION            |\n");
    printf("\t\t ---------------------------------------------\n");
    fflush(stdout);

    double *Foods, *f, *fitness, *trial, *prob, *solution, *GlobalParams, *GlobalMins,
            sommaTempo, tempoMedio, GlobalMin, mean, std, errorCriteria;
    int n, cycle, i, j, numofdims, colonySize, FoodNumber, maxCycle, limit, runtime;
    time_t start, stop;
    FILE *fd_ini, *fd_results_bees, *fd_results_global, *fd_results_minbees;
    char *file_results_bees, *file_results_global, *file_results_minbees;

    /* Allocate mem for filenames */
    if ((file_results_bees = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_bees)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_minbees = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_minbees)\n");
        return (-1);
    }

    /*read ini*/
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
    fscanf(fd_ini, "colonySize=%d\n", &colonySize);

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

    fscanf(fd_ini, "limit=%d\n", &limit);
    fscanf(fd_ini, "errorCriteria=%lf\n", &errorCriteria);
    fscanf(fd_ini, "runtime=%d\n", &runtime);
    fscanf(fd_ini, "maxCycle=%d\n", &maxCycle);
    fflush(fd_ini);
    fclose(fd_ini);

    FoodNumber = ((colonySize % 2) == 0) ? (colonySize / 2) : round(colonySize / 2) + 1;

    /* allocateMem */
    printf("Allocating memory\n");
    fflush(stdout);
    if ((Foods = (double *) calloc(FoodNumber * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*Foods)\n");
        return (-1);
    }
    if ((GlobalParams = (double *) calloc(numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*GlobalParams)\n");
        return (-1);
    }
    if ((f = (double *) calloc(FoodNumber, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*f)\n");
        return (-1);
    }
    if ((fitness = (double *) calloc(FoodNumber, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*fitness)\n");
        return (-1);
    }
    if ((trial = (double *) calloc(FoodNumber, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*trial)\n");
        return (-1);
    }
    if ((prob = (double *) calloc(FoodNumber, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*prob)\n");
        return (-1);
    }
    if ((solution = (double *) calloc(numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*solution)\n");
        return (-1);
    }
    if ((GlobalMins = (double *) calloc(runtime, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*GlobalMins)\n");
        return (-1);
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_bees, 35, "results_bees_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 35, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_minbees, 35, "results_minbees_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_bees = fopen(file_results_bees, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_minbees = fopen(file_results_minbees, "wb+");

    /* write headers into files */
    // results for bees
    fprintf(fd_results_bees, "#Runtime,#Cycle,#Bee");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_bees, ",Dim[%d]", i);
    }
    fprintf(fd_results_bees, "\n");
    // results for minbees
    fprintf(fd_results_minbees, "#Runtime,#Cycle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_minbees, ",Dim[%d]", i);
    }
    fprintf(fd_results_minbees, ",GlobalMin,Error\n");

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    mean = 0;
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);
        fflush(stdout);

        printf("\tInitializing food sources\n");
        fflush(stdout);
        init(FoodNumber, numofdims, lb, ub, Foods, solution, f, fitness, &GlobalMin, GlobalParams);

        MemorizeBestSource(FoodNumber, numofdims, f, Foods, GlobalParams, &GlobalMin);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            SendEmployedBees(FoodNumber, numofdims, lb, ub, Foods, solution, fitness, trial, f);
            CalculateProbabilities(FoodNumber, fitness, prob);
            SendOnlookerBees(FoodNumber, numofdims, lb, ub, Foods, solution, fitness, trial, f, prob);
            MemorizeBestSource(FoodNumber, numofdims, f, Foods, GlobalParams, &GlobalMin);
            SendScoutBees(FoodNumber, numofdims, lb, ub, Foods, solution, f, fitness, trial, limit, &GlobalMin, GlobalParams);

            /* saving results into files */
            // bees
            for (i = 0; i < FoodNumber; i++) {
                fprintf(fd_results_bees, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < numofdims; j++) {
                    fprintf(fd_results_bees, ",%4.3f", Foods[i * numofdims + j]);
                }
                fprintf(fd_results_bees, "\n");
            }
            // minbees
            fprintf(fd_results_minbees, "%d,%d", n + 1, cycle);
            for (i = 0; i < numofdims; i++) {
                fprintf(fd_results_minbees, ",%4.3f", GlobalParams[i]);
            }
            fprintf(fd_results_minbees, ",%4.3f,%4.3f\n", GlobalMin, fitness[0]);

            cycle++;
        } while ((cycle <= maxCycle) && (fitness[0] > errorCriteria));

        GlobalMins[n] = GlobalMin;
        mean += GlobalMin;

        time(&stop);
        printf("\n\tElapsed time: %.3f s - GlobalMin: %4.3f\n", difftime(stop, start), GlobalMin);
        fflush(stdout);

        // save info into global files
        fprintf(fd_results_global, "Runtime #%d, elapsed time %.3f s, GlobalMin: %4.3f\n",
                n + 1, difftime(stop, start), GlobalMin);

        sommaTempo += difftime(stop, start);
    }

    mean = mean / runtime;
    tempoMedio = sommaTempo / runtime;
    std = 0;
    for (i = 0; i < runtime; i++) {
        std += (mean - GlobalMins[i]) * (mean - GlobalMins[i]);
    }
    std = sqrt(std / runtime);

    printf("Means of GlobalMin of %d runs: %4.3f +/- %4.3f\nAverage optimization time: %.3f s\n", runtime, mean, std, tempoMedio);
    fflush(stdout);

    // save info into global files
    fprintf(fd_results_global, "Means of GlobalMin of %d runs +/- %4.3f: %4.3f\nAverage optimization time: %.3f s\n", runtime, mean, std, tempoMedio);

    fflush(fd_results_bees);
    fclose(fd_results_bees);
    fflush(fd_results_minbees);
    fclose(fd_results_minbees);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(file_results_bees);
    free(file_results_minbees);
    free(file_results_global);
    free(Foods);
    free(GlobalParams);
    free(f);
    free(fitness);
    free(trial);
    free(prob);
    free(solution);
    free(GlobalMins);

    return (0);
}