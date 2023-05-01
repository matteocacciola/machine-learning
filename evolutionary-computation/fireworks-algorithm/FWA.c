#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <string.h>
#include <omp.h>
#include "function.h"

//constant variables
#define PI (3.14159265358979323846264338327950288)
#define EPS (1e-38)
#define min(X,Y)        ((X) < (Y) ? (X) : (Y))
#define max(X,Y)        ((X) > (Y) ? (X) : (Y))
#define sign(x)         (( x > 0 ) - ( x < 0 ))

double CalculateFitness(double);
double evaluate(double *, int, double, double);
double lea(double, double);
double randn(double, double);
double minArray(double *, int, int *);
void init(double *, double *, int, int, double *, double *, double, double);
int computeSparksForEachFirework(int, int, int, int, int, double, double *, int *);
void computeAmplitudeExplosions(int, int, double, double *, double *, double *, double *, double *, double *, double *);
void locateSpecificSpark(int, int, int, int *, double *, double *, double *, double *, double *, double, double);
void locateSpark(int, int, int, int *, double *, int *, double *, double *, double *, double *, double *, double, double);
void findMinimaLocations(int, int, int, double *, double *, double *, double *, double *, double *);

/**
 * Min of an array. The function also returns the index of the maximum value
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
#pragma omp parallel shared(minimum, indx, array, N) firstprivate(minimumLocal, indxLocal)
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
 * @param m
 * @param dimension
 * @param center
 * @param f_bias
 * @return 
 */
double evaluate(double *m, int dimension, double center, double f_bias) {
    int i;
    double rs;

#pragma omp parallel for shared(m, dimension, center) private(i)
    for (i = 0; i < dimension; i++) {
        m[i] -= center;
    }

    rs = function(m, dimension) + f_bias;

#pragma omp parallel for shared(m, dimension, center) private(i)
    for (i = 0; i < dimension; i++) {
        m[i] += center;
    }

    return rs;
}

/**
 * 
 * @param a
 * @param b
 * @return 
 */
double lea(double a, double b) {
    srand(time(NULL));
    return a + (b - a)*(rand() / (double) RAND_MAX);
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
 * 
 * @param fireworks
 * @param fitnesses
 * @param D
 * @param N
 * @param lb
 * @param ub
 * @param center
 * @param bias
 */
void init(double *fireworks, double *fitnesses, int D, int N, double *lb, double *ub, double center, double bias) {
    double temp[D];
    int i, j;

    /* init fireworks and calculate fitness for each set of possible solutions */
#pragma omp parallel for shared(fitnesses, fireworks, N, D, center, bias, lb, ub) private(i, j, temp)
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            double dummy = lea(lb[j], ub[j]);
            fireworks[i * D + j] = dummy;
            temp[j] = dummy;
        }
        fitnesses[i] = evaluate(temp, D, center, bias);
    }
}

/**
 * 
 * @param N
 * @param MM
 * @param M
 * @param AM
 * @param BM
 * @param maxVal
 * @param fitnesses
 * @param nums
 * @return 
 */
int computeSparksForEachFirework(int N, int MM, int M, int AM, int BM, double maxVal, double *fitnesses, int *nums) {
    int sum = 0;
    int i, temp;
#pragma omp parallel for shared(N, maxVal, fitnesses) private(i) reduction(+:sum)
    for (i = 0; i < N; i++) {
        sum += (maxVal - fitnesses[i]);
    }

    int counter = N + MM;
#pragma omp parallel for shared(nums, M, maxVal, fitnesses, BM, AM) private(i, temp) reduction (+:counter)
    for (i = 0; i < N; i++) {
        temp = (int) (M * ((maxVal - fitnesses[i] + EPS) / (sum + EPS)));
        temp = min(max(temp, BM), AM);
        nums[i] = temp;
        counter += nums[i];
    }

    return counter;
}

/**
 * 
 * @param N
 * @param D
 * @param minVal
 * @param lb
 * @param ub
 * @param As
 * @param fitnesses
 * @param positions
 * @param fireworks
 * @param vals
 */
void computeAmplitudeExplosions(int N, int D, double minVal, double *lb, double *ub, double *As, double *fitnesses, double *positions, double *fireworks, double *vals) {
    int sum = 0;
    int i, idx, j;
#pragma omp parallel for shared(N, minVal, fitnesses) private(i) reduction(+:sum)
    for (i = 0; i < N; i++) {
        sum += (fitnesses[i] - minVal);
    }

#pragma omp parallel for shared(As, N, ub, lb, fitnesses, minVal, sum) private(i)
    for (i = 0; i < N; i++) {
        double ampSpark = ((ub[i] - lb[i]) / 5.0);
        As[i] = ampSpark * ((fitnesses[i] - minVal + EPS) / (sum + EPS));
    }
#pragma omp parallel for shared(positions, vals, N, D, fitnesses, fireworks) private(idx, j)
    for (idx = 0; idx < N; idx++) {
        for (j = 0; j < D; j++) {
            positions[idx * D + j] = fireworks[idx * D + j];
        }
        vals[idx] = fitnesses[idx];
    }
}

/**
 * 
 * @param MM
 * @param D
 * @param flags
 * @param lb
 * @param ub
 * @param fireworks
 * @param positions
 * @param vals
 */
void locateSpecificSpark(int MM, int D, int N, int *flags, double *lb, double *ub, double *fireworks, double *positions, double *vals, double center, double bias) {
    double temp[D], g, lbJ, ubJ, range, tt;
    int idx = N - 1;
    int i, j, nn, count, index, id;

    srand(time(NULL));
#pragma omp parallel for shared(positions, vals, MM, D, flags, fireworks, lb, ub, idx, center, bias) private(i, j, nn, count, index, id, g, lbJ, ubJ, range, tt, temp)
    for (i = 0; i < MM; i++) {
        memset(flags, 0, sizeof (int) * D);
        nn = rand() % D + 1;
        count = 0;
        while (count < nn) {
            index = rand() % D;
            if (flags[index] != 1) {
                flags[index] = 1;
                count++;
            }
        }

        id = rand() % N;
        g = randn(1, 1);
        for (j = 0; j < D; j++) {
            if (flags[j] == 1) {
                positions[idx * D + j] = fireworks[id * D + j] * g;
                lbJ = lb[j];
                ubJ = ub[j];
                range = ubJ - lbJ;
                if (positions[idx * D + j] < lbJ) {
                    tt = fabs(positions[idx * D + j]);
                    while (tt > 0) {
                        tt -= range;
                    }
                    tt += range;
                    positions[idx * D + j] = lbJ + tt;
                } else if (positions[idx * D + j] > ubJ) {
                    tt = fabs(positions[idx * D + j]);
                    while (tt > 0) {
                        tt -= range;
                    }
                    tt += range;
                    positions[idx * D + j] = lbJ + tt;
                }
            } else {
                positions[idx * D + j] = fireworks[id * D + j];
            }
            temp[j] = positions[idx * D + j];
        }
        vals[idx] = evaluate(temp, D, center, bias);
        idx++;
    }
}

/**
 * 
 * @param N
 * @param D
 * @param MM
 * @param nums
 * @param As
 * @param flags
 * @param lb
 * @param ub
 * @param fireworks
 * @param positions
 * @param vals
 * @param center
 * @param bias
 */
void locateSpark(int N, int D, int MM, int *nums, double *As, int *flags, double *lb, double *ub, double *fireworks, double *positions, double *vals, double center, double bias) {
    int i, j, k, nn, count, numsI, index;
    int idx = N + MM - 1;
    double temp[D], tt, lbJ, ubJ, h;

    srand(time(NULL));
#pragma omp parallel for shared(positions, vals, N, D, nums, As, flags, fireworks, lb, ub, idx, center, bias) private(i, j, k, nn, count, numsI, index, tt, lbJ, ubJ, h, temp)
    for (i = 0; i < N; i++) {
        numsI = nums[i];
        for (k = 0; k < numsI; k++, idx++) {
            memset(flags, 0, sizeof (int) * D);
            nn = rand() % D + 1;
            count = 0;
            while (count < nn) {
                index = rand() % D;
                if (flags[index] != 1) {
                    flags[index] = 1;
                    count++;
                }
            }

            for (j = 0; j < D; j++) {
                lbJ = lb[j];
                ubJ = ub[j];
                h = As[i] * lea(lbJ, ubJ);
                if (flags[j] == 1) {
                    positions[idx * D + j] = fireworks[i * D + j] + h;
                    double range = ubJ - lbJ;
                    if (positions[idx * D + j] < lbJ) {
                        tt = fabs(positions[idx * D + j]);
                        while (tt > 0) {
                            tt -= range;
                        }
                        tt += range;
                        positions[idx * D + j] = lbJ + tt;
                    } else if (positions[idx * D + j] > ubJ) {
                        tt = fabs(positions[idx * D + j]);
                        while (tt > 0) {
                            tt -= range;
                        }
                        tt += range;
                        positions[idx * D + j] = lbJ + tt;
                    }
                } else {
                    positions[idx * D + j] = fireworks[i * D + j];
                }
                temp[j] = positions[idx * D + j];
            }
            vals[idx] = evaluate(temp, D, center, bias);
        }
    }
}

/**
 * find the minima
 * 
 * @param N
 * @param counter
 * @param D
 * @param ub
 * @param vals
 * @param fitnesses
 * @param positions
 * @param fireworks
 * @param Rs
 */
void findMinimaLocations(int N, int counter, int D, double *ub, double *vals, double *fitnesses, double *positions, double *fireworks, double *Rs) {
    int index;
    int i, j, k, id;
    double r = 0;
    double tem, s, p;
    
    fitnesses[0] = minArray(vals, counter, &index);

#pragma omp parallel for shared(fireworks, D, positions, index) private(i) reduction(+:r)
    for (i = 0; i < D; i++) {
        fireworks[i] = positions[index * D + i];
        r += pow(lea(0, ub[i]), 2);
    }
    r = sqrt(r) / D;

    double ss = 0;
#pragma omp parallel for shared(Rs, positions, counter, D) private(i, j, k, tem) reduction(+:ss)
    for (i = 0; i < counter; i++) {
        for (j = 0; j < counter; j++) {
            tem = 0;
            for (k = 0; k < D; k++) {
                tem += (positions[i * D + k] - positions[j * D + k])*(positions[i * D + k] - positions[j * D + k]);
            }
            Rs[i] += sqrt(tem);
        }
        ss += Rs[i];
    }

#pragma omp parallel for shared(Rs, ss, counter) private(i)
    for (i = 0; i < counter; i++) {
        Rs[i] = Rs[i] / ss;
    }

#pragma omp parallel for shared(N, ub, counter, Rs, D, fireworks, positions, fitnesses, vals, r) private(i, j, id, p, s)
    for (i = 1; i < N; i++) {
        s = 0;
        p = 0;
        for (id = 0; id < counter && p <= r; id++) {
            s += pow(Rs[id], 2);
            p = sqrt(s);
        }
        for (j = 0; j < D; j++) {
            fireworks[i * D + j] = positions[id * D + j];
        }
        fitnesses[i] = vals[id];
    }
}

/**
 * Main program of FWA (Fireworks Algorithm)
 * 
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char **argv) {
    printf("\t\t ---------------------------------------------\n");
    printf("\t\t|         WELCOME TO THE PROGRAM FOR          |\n");
    printf("\t\t|           FIREWORKS OPTIMIZATION            |\n");
    printf("\t\t ---------------------------------------------\n");
    fflush(stdout);

    double *fireworks, *fitnesses, *positions, *vals, *Rs, *As, *minFits, sumRs,
            errorCriteria, actError, sommaTempo, mean, std, tempoMedio,
            center, f_bias, A, B, minVal, maxVal, fitnessZeroUnbiased;
    int *nums, *flags, numofdims, numFireworks, numSparks, numGaussianSparks, runtime, maxCycle, cycle,
            AM, BM, buffer, i, j, n, counter;
    time_t start, stop;
    FILE *fd_ini, *fd_results_fireworks, *fd_results_global, *fd_results_minfireworks;
    char *file_results_fireworks, *file_results_global, *file_results_minfireworks;

    /* Allocate mem for filenames */
    if ((file_results_fireworks = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_fireworks)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_minfireworks = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_minfireworks)\n");
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
    fscanf(fd_ini, "N=%d\n", &numFireworks);
    fscanf(fd_ini, "M=%d\n", &numSparks);
    fscanf(fd_ini, "A=%lf\n", &A);
    fscanf(fd_ini, "B=%lf\n", &B);
    fscanf(fd_ini, "MM=%d\n", &numGaussianSparks);

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

    fscanf(fd_ini, "center=%lf\n", &center);
    fscanf(fd_ini, "f_bias=%lf\n", &f_bias);
    fscanf(fd_ini, "errorCriteria=%lf\n", &errorCriteria);
    fscanf(fd_ini, "runtime=%d\n", &runtime);
    fscanf(fd_ini, "maxCycle=%d\n", &maxCycle);
    fflush(fd_ini);
    fclose(fd_ini);

    AM = (int) (A * numSparks);
    BM = (int) (B * numSparks);
    buffer = numSparks + numGaussianSparks + numSparks * BM;
    sumRs = 0;

    /* allocateMem */
    printf("Allocating memory\n");
    fflush(stdout);
    if ((fireworks = (double *) calloc(numFireworks * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*fireworks)\n");
        return (-1);
    }
    if ((fitnesses = (double *) calloc(numFireworks, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*fitnesses)\n");
        return (-1);
    }
    if ((positions = (double *) calloc(buffer * numofdims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*positions)\n");
        return (-1);
    }
    if ((vals = (double *) calloc(buffer, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*vals)\n");
        return (-1);
    }
    if ((As = (double *) calloc(numFireworks, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*As)\n");
        return (-1);
    }
    if ((Rs = (double *) calloc(buffer, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*Rs)\n");
        return (-1);
    }
    if ((nums = (int *) calloc(numFireworks, sizeof (int))) == NULL) {
        printf("ERROR!!! Not enough memory (*nums)\n");
        return (-1);
    }
    if ((flags = (int *) calloc(numofdims, sizeof (int))) == NULL) {
        printf("ERROR!!! Not enough memory (*flags)\n");
        return (-1);
    }
    if ((minFits = (double *) calloc(runtime, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*minFits)\n");
        return (-1);
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_fireworks, 40, "results_fireworks_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 40, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_minfireworks, 40, "results_minfireworks_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_fireworks = fopen(file_results_fireworks, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_minfireworks = fopen(file_results_minfireworks, "wb+");

    /* write headers into files */
    // results for fireworks
    fprintf(fd_results_fireworks, "#Runtime,#Cycle,#Firework");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_fireworks, ",Dim[%d]", i);
    }
    fprintf(fd_results_fireworks, "\n");
    // results for minfireworks
    fprintf(fd_results_minfireworks, "#Runtime,#Cycle");
    for (i = 0; i < numofdims; i++) {
        fprintf(fd_results_minfireworks, ",Dim[%d]", i);
    }
    fprintf(fd_results_minfireworks, ",MinFit,Error\n");

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    mean = 0;
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);
        fflush(stdout);

        printf("\tInitializing fireworks\n");
        fflush(stdout);
        init(fireworks, fitnesses, numofdims, numFireworks, lb, ub, center, f_bias);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            minVal = maxVal = fitnesses[0];
            for (i = 1; i < numFireworks; i++) {
                minVal = min(minVal, fitnesses[i]);
                maxVal = max(maxVal, fitnesses[i]);
            }

            /* compute total number of sparks generated by each firework */
            counter = computeSparksForEachFirework(numFireworks, numGaussianSparks, numSparks, AM, BM, maxVal, fitnesses, nums);

            /* compute amplitudes of explosions */
            computeAmplitudeExplosions(numFireworks, numofdims, minVal, lb, ub, As, fitnesses, positions, fireworks, vals);

            /* locate specific (Gaussian) sparks */
            locateSpecificSpark(numGaussianSparks, numofdims, numFireworks, flags, lb, ub, fireworks, positions, vals, center, f_bias);

            /* locate sparks */
            //locateSpark(numFireworks, numofdims, numGaussianSparks, nums, As, flags, lb, ub, fireworks, positions, vals, center, f_bias);

            /* find minima locations */
            findMinimaLocations(numFireworks, counter, numofdims, ub, vals, fitnesses, positions, fireworks, Rs);

            fitnessZeroUnbiased = fitnesses[0] - f_bias;
            actError = CalculateFitness(fitnessZeroUnbiased);

            /* saving results into files */
            // fireworks
            for (i = 0; i < numFireworks; i++) {
                fprintf(fd_results_fireworks, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < numofdims; j++) {
                    fprintf(fd_results_fireworks, ",%4.3f", fireworks[i * numofdims + j]);
                }
                fprintf(fd_results_fireworks, "\n");
            }
            // minfireworks
            fprintf(fd_results_minfireworks, "%d,%d", n + 1, cycle);
            for (i = 0; i < numofdims; i++) {
                fprintf(fd_results_minfireworks, ",%4.3f", fireworks[i]);
            }
            fprintf(fd_results_minfireworks, ",%4.3f,%4.3f\n", fitnessZeroUnbiased, actError);

            /* showing something to prompt */
            // printf("\t\tCycle %3d\tMinFit: %4.3f\n", cycle, fitnesses[0]);

            cycle++;
        } while ((cycle <= maxCycle) && (actError > errorCriteria));

        minFits[n] = fitnessZeroUnbiased;
        sumRs += minFits[n];

        time(&stop);
        printf("\n\tElapsed time: %.3f s - MinFit: %4.3f\n", difftime(stop, start), minFits[n]);
        fflush(stdout);

        // save info into global files
        fprintf(fd_results_global, "Runtime #%d, elapsed time %.3f s, MinFit: %4.3f\n",
                n + 1, difftime(stop, start), minFits[n]);

        sommaTempo += difftime(stop, start);
    }

    mean = sumRs / runtime;
    tempoMedio = sommaTempo / runtime;
    std = 0;
    for (i = 0; i < runtime; i++) {
        std += (mean - minFits[i]) * (mean - minFits[i]);
    }
    std = sqrt(std / runtime);

    printf("Means of MinFit of %d runs: %4.3f +/- %4.3f\nAverage optimization time: %.3f s\n", runtime, mean, std, tempoMedio);
    fflush(stdout);
    
    // save info into global files
    fprintf(fd_results_global, "Means of MinFit of %d runs: %4.3f +/- %4.3f\nAverage optimization time: %.3f s\n", runtime, mean, std, tempoMedio);

    fflush(fd_results_fireworks);
    fclose(fd_results_fireworks);
    fflush(fd_results_minfireworks);
    fclose(fd_results_minfireworks);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(file_results_fireworks);
    free(file_results_minfireworks);
    free(file_results_global);
    free(fireworks);
    free(fitnesses);
    free(positions);
    free(vals);
    free(As);
    free(Rs);
    free(nums);
    free(flags);
    free(minFits);

    return (0);
}