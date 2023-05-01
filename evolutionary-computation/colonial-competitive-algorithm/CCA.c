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
#define min(x,y)        ((x) < (y) ? (x) : (y))
#define max(x,y)        ((x) > (y) ? (x) : (y))

double CalculateFitness(double);
double sumArray(double *, int);
double meanArray(double *, int);
double maxArray(double *, int, int *);
double minArray(double *, int, int *);
void randperm(int, int *);
void sortCountriesOnCosts(int, int, double *, double *);
void initCountries(int, int, double *, double *, double *, double *);
double generateNewCountryPosCoord(double, double);
void initEmpires(int, int, int, double *, double *, double *, double **, int *);
void initColoniesOfEmpires(int, int, int, int, double, double *, double *, double *, double *, double *, double *, int *, double **, double **);
void assimilateColonies(int, int, double *, double *, double *, double *, double);
void revolveColonies(int, double, double *, double *, int, double *);
void evalNewColoniesPosition(int, int, double *, double *);
void possessEmpire(int, double, int, double *, double *, double *);
void removeEmpire(double **, double **, int **, double ***, double ***, double ***, int, int *, int);
void uniteSimilarEmpires(int, int *, double *, double *, double, double, double **, double **, int **, double ***, double ***, double ***);
int selectAnEmpire(double *, int);
void imperialisticCompetition(int, int *, double **, double **, int **, double ***, double ***, double ***);

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
 * Mean of an array
 * 
 * @param array
 * @param N
 * @return 
 */
double meanArray(double *array, int N) {
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
#pragma omp parallel shared(maximum, indx, array, N) firstprivate(maximumLocal, indxLocal)
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
 * Random permutation.
 * P = randperm(N) returns a N-size vector containing a random permutation of the
 * integers 1:N
 * 
 * @param n
 * @param r
 */
void randperm(int n, int *r) {
    int i;

    for (i = 0; i < n; i++) {
        r[i] = i;
    }

    srand(time(NULL));
    // Fisher–Yates shuffle algorithm
    for (i = 0; i < n; i++) {
        int j = n * ((int) rand() / ((int) (RAND_MAX) + 1.));
        // j = rand() % (n - i) + i;
        int temp = r[j];
        r[j] = r[i];
        r[i] = temp;
    }
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
void sortCountriesOnCosts(int n, int ndims, double *positions, double *costs) {
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
 * Initialize countries
 * 
 * @param N
 * @param D
 * @param lb
 * @param ub
 * @param countries
 */
void initCountries(int N, int D, double *lb, double *ub, double *countries, double *costs) {
    int i, j;
    double temp[D], dummy, lbJ, ubJ, scale;

    srand(time(NULL));
#pragma omp parallel for shared(countries, costs, N, D, lb, ub) private(i, j, dummy, temp, lbJ, ubJ, scale)
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            lbJ = lb[j];
            scale = fabs(ub[j] - lbJ);
            dummy = lbJ + (scale * ((double) rand() / ((double) (RAND_MAX) + 1.)));
            countries[i * D + j] = dummy;
            temp[j] = dummy;
        }
        costs[i] = function(temp, D);
    }

    /* sorting initial countries */
    sortCountriesOnCosts(N, D, countries, costs);
}

/**
 * Generate new country
 * 
 * @param lb
 * @param ub
 */
double generateNewCountryPosCoord(double lb, double ub) {
    double res;

    srand(time(NULL));
    res = lb + ((ub - lb) * ((double) rand() / ((double) (RAND_MAX) + 1.)));

    return res;
}

/**
 * Create the initial Empires
 * 
 * @param NI
 * @param D
 * @param NAC
 * @param countries
 * @param countryCosts
 * @param imperialistCosts
 * @param imperialistPos
 * @param nColonies
 */
void initEmpires(int NI, int D, int NAC, double *countries, double *countryCosts, double *imperialistCosts, double **imperialistPos, int *nColonies) {
    int i, j, dummyInt, sumAINC;
    double imperialistPowers[NI], dummyDbl, sumAIP;

    /* copy first NI countries as empires */
#pragma omp parallel for shared(imperialistPos, imperialistCosts, NI, D, countryCosts, countries) private(i, j)
    for (i = 0; i < NI; i++) {
        for (j = 0; j < D; j++) {
            imperialistPos[i][j] = countries[i * D + j];
        }
        imperialistCosts[i] = countryCosts[i];
    }

    /* calculate power of imperialists */
    int dummy;
    dummyDbl = maxArray(imperialistCosts, NI, &dummy);
    sumAIP = 0;
#pragma omp parallel for shared(imperialistCosts, imperialistPowers, NI, D, dummyDbl) private(i) reduction(+:sumAIP)
    for (i = 0; i < NI; i++) {
        imperialistPowers[i] = (dummyDbl > 0)
                ? (1.3 * dummyDbl - imperialistCosts[i])
                : (0.7 * dummyDbl - imperialistCosts[i]);
        sumAIP += imperialistPowers[i];
    }

    /* calculate number of colonies for all imperialists */
    sumAINC = 0;
#pragma omp parallel for shared(imperialistCosts, imperialistPowers, NI, D, sumAIP, NAC) private(i, dummyInt) reduction(+:sumAINC)
    for (i = 0; i < (NI - 1); i++) {
        dummyInt = (int) round(imperialistPowers[i] / sumAIP * NAC);
        nColonies[i] = dummyInt;
        sumAINC += dummyInt;
    }
    nColonies[NI - 1] = NAC - sumAINC;
}

/**
 * Create the initial colonies of the Empires
 * 
 * @param NI
 * @param N
 * @param D
 * @param NAC
 * @param zeta
 * @param lb
 * @param ub
 * @param countries
 * @param countryCosts
 * @param imperialistCosts
 * @param totalCosts
 * @param nColonies
 * @param coloniesPos
 * @param coloniesCost
 */
void initColoniesOfEmpires(int NI, int N, int D, int NAC, double zeta, double *lb, double *ub, double *countries, double *countryCosts, double *imperialistCosts, double *totalCosts, int *nColonies, double **coloniesPos, double **coloniesCost) {
    int i, j, k, *RandomIndexes, indHigh, indx;
    double AllColoniesPosition[((N - NI) * D)], AllColoniesCost[(N - NI)], v;
    
    /* copy remaining countries as colonies */
#pragma omp parallel for shared(AllColoniesPosition, AllColoniesCost, RandomIndexes, N, D, NI, countries, countryCosts, NAC) private(i, j)
    for (i = NI; i < N; i++) {
        for (j = 0; j < D; j++) {
            AllColoniesPosition[(i - NI) * D + j] = countries[i * D + j];
        }
        AllColoniesCost[(i - NI)] = countryCosts[i];
    }

    if ((RandomIndexes = (int *) calloc(NAC, sizeof (int))) == NULL) {
        printf("ERROR!!! Not enough memory (initColoniesOfEmpires->*RandomIndexes)\n");
        exit(-1);
    }
    randperm(NAC, RandomIndexes);

    /* create the empires */
#pragma omp parallel for shared(nColonies, coloniesPos, coloniesCost, totalCosts, NI, D, RandomIndexes, AllColoniesPosition, AllColoniesCost, imperialistCosts, zeta) private(i, j, indHigh, indx, v)
    for (i = 0; i < NI; i++) {
        if (nColonies[i] > 0) {
            indHigh = nColonies[i];

            for (j = 0; j < indHigh; j++) {
                indx = RandomIndexes[j];
                for (k = 0; k < D; k++) {
                    coloniesPos[i][(j * D) + k] = AllColoniesPosition[(indx * D) + k];
                }
                coloniesCost[i][j] = AllColoniesCost[indx];
            }
        } else {
            for (j = 0; j < D; j++) {
                coloniesPos[i][j] = generateNewCountryPosCoord(lb[j], ub[j]);
            }
            v = function(coloniesPos[i], D);
            coloniesCost[i] = &v;
            nColonies[i] = 1;
        }
        totalCosts[i] = (nColonies[i] > 0)
                ? imperialistCosts[i] + zeta * meanArray(coloniesCost[i], indHigh)
                : imperialistCosts[i] + zeta * coloniesCost[i][0];
    }

    free(RandomIndexes);
}

/**
 * Assimilate colonies
 * 
 * @param D
 * @param nColonies
 * @param imperialistPos
 * @param coloniesPos
 * @param lb
 * @param ub
 * @param assimCoeff
 */
void assimilateColonies(int D, int nColonies, double *imperialistPos, double *coloniesPos, double *lb, double *ub, double assimCoeff) {
    int i, j;
    double diffPos;

    srand(time(NULL));
#pragma omp parallel for shared(coloniesPos, nColonies, D, imperialistPos, assimCoeff, lb, ub) private(i, j, diffPos)
    for (i = 0; i < nColonies; i++) {
        for (j = 0; j < D; j++) {
            diffPos = imperialistPos[j] - coloniesPos[(i * D) + j];
            coloniesPos[(i * D) + j] += min(max(2 * assimCoeff * diffPos * ((double) rand() / ((double) (RAND_MAX) + 1.)), lb[j]), ub[j]);
        }
    }
}

/**
 * Revolve colonies
 * 
 * @param D
 * @param revRate
 * @param lb
 * @param ub
 * @param nColonies
 * @param coloniesPos
 */
void revolveColonies(int D, double revRate, double *lb, double *ub, int nColonies, double *coloniesPos) {
    int i, j, nRevColonies, indx, *r;

    nRevColonies = (int) round(revRate * nColonies);

    if ((r = (int *) calloc(nRevColonies, sizeof (int))) == NULL) {
        printf("ERROR!!! Not enough memory (revolveColonies->*r)\n");
        exit(-1);
    }
    randperm(nRevColonies, r);

#pragma omp parallel for shared(coloniesPos, nRevColonies, D, lb, ub) private(i, j, indx)
    for (i = 0; i < nRevColonies; i++) {
        indx = r[i];
        for (j = 0; j < D; j++) {
            coloniesPos[(indx * D) + j] = generateNewCountryPosCoord(lb[j], ub[j]);
        }
    }

    free(r);
}

/**
 * New Cost Evaluation
 * 
 * @param D
 * @param nColonies
 * @param coloniesPos
 * @param coloniesCost
 */
void evalNewColoniesPosition(int D, int nColonies, double *coloniesPos, double *coloniesCost) {
    int i, j;
    double temp[D];

#pragma omp parallel for shared(coloniesCost, nColonies, D, coloniesPos) private(i, j, temp)
    for (i = 0; i < nColonies; i++) {
        for (j = 0; j < D; j++) {
            temp[j] = coloniesPos[(i * D) + j];
        }
        coloniesCost[i] = function(temp, D);
    }
}

/**
 * Empire Possession (++++++ Power Possession, Empire Possession)
 * 
 * @param D
 * @param imperialistCosts
 * @param nColonies
 * @param imperialistPos
 * @param coloniesPos
 * @param coloniesCost
 */
void possessEmpire(int D, double imperialistCost, int nColonies, double *imperialistPos, double *coloniesPos, double *coloniesCost) {
    int i, indx, minCost;
    double OldImperialistPosition[D], OldImperialistCost;

    minCost = minArray(coloniesCost, nColonies, &indx);
    if (minCost < imperialistCost) {
#pragma omp parallel for shared(OldImperialistPosition, OldImperialistCost, D, imperialistPos) private(i)
        for (i = 0; i < D; i++) {
            OldImperialistPosition[i] = imperialistPos[i];
        }
        OldImperialistCost = imperialistCost;
#pragma omp parallel for shared(coloniesPos, imperialistPos, D, OldImperialistPosition) private(i)
        for (i = 0; i < D; i++) {
            imperialistPos[i] = coloniesPos[(indx * D) + i];
            coloniesPos[(indx * D) + i] = OldImperialistPosition[i];
        }
        imperialistCost = coloniesCost[indx];
        coloniesCost[indx] = OldImperialistCost;
    }
}

/**
 * 
 * @param imperialistCosts
 * @param totalCosts
 * @param nColonies
 * @param imperialistPos
 * @param coloniesPos
 * @param coloniesCost
 * @param index
 * @param nEmp
 * @param D
 */
void removeEmpire(double **imperialistCosts, double **totalCosts, int **nColonies, double ***imperialistPos, double ***coloniesPos, double ***coloniesCost, int index, int *nEmp, int D) {
    if ((*nEmp > 0) && (index < *nEmp) && (index >= 0)) {
        int i, dummy;
        *nEmp -= 1;

        double *tempIC, *tempTC, **tempIP, **tempCP, **tempCC;
        int *tempNC;

        int nE = *nEmp;
        if ((tempIC = (double *) calloc(nE, sizeof (double))) == NULL) {
            printf("ERROR!!! Not enough memory (removeEmpire->*tempIC)\n");
            exit(-1);
        }
        if ((tempTC = (double *) calloc(nE, sizeof (double))) == NULL) {
            printf("ERROR!!! Not enough memory (removeEmpire->*tempTC)\n");
            exit(-1);
        }
        if ((tempNC = (int *) calloc(nE, sizeof (int))) == NULL) {
            printf("ERROR!!! Not enough memory (removeEmpire1->tempNC)\n");
            exit(-1);
        }

        if ((tempIP = (double **) calloc(nE, sizeof (double *))) == NULL) {
            printf("ERROR!!! Not enough memory (removeEmpire->**tempIP)\n");
            exit(-1);
        }
        if ((tempCP = (double **) calloc(nE, sizeof (double *))) == NULL) {
            printf("ERROR!!! Not enough memory (removeEmpire->**tempCP)\n");
            exit(-1);
        }
        if ((tempCC = (double **) calloc(nE, sizeof (double *))) == NULL) {
            printf("ERROR!!! Not enough memory (removeEmpire->**tempCC)\n");
            exit(-1);
        }

        /* copy everything BEFORE the index into the temp arrays */
        memmove(tempIC, *imperialistCosts, index * sizeof (double));
        memmove(tempTC, *totalCosts, index * sizeof (double));
        memmove(tempNC, *nColonies, index * sizeof (int));
#pragma omp parallel for shared(tempIP, tempCP, tempCC, index, nColonies, D, imperialistPos, coloniesPos, coloniesCost) private(i, dummy)
        for (i = 0; i < index; i++) {
            dummy = (*nColonies)[i];

            if ((tempIP[i] = (double *) calloc(D, sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (removeEmpire->(*tempIP)[%d])\n", i);
                exit(-1);
            }
            memmove(tempIP[i], (*imperialistPos)[i], D * sizeof (double));

            if ((tempCP[i] = (double *) calloc(dummy * D, sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (removeEmpire->*tempCP[%d])\n", i);
                exit(-1);
            }
            memmove(tempCP[i], (*coloniesPos)[i], dummy * D * sizeof (double));

            if ((tempCC[i] = (double *) calloc(dummy, sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (removeEmpire->*tempCC[%d])\n", i);
                exit(-1);
            }
            memmove(tempCC[i], (*coloniesCost)[i], dummy * sizeof (double));
        }

        /* copy everything AFTER the index into the temp arrays */
        memmove(tempIC + index, (*imperialistCosts) + (index + 1), (nE - index) * sizeof (double));
        memmove(tempTC + index, (*totalCosts) + (index + 1), (nE - index) * sizeof (double));
        memmove(tempNC + index, (*nColonies) + (index + 1), (nE - index) * sizeof (int));
#pragma omp parallel for shared(tempIP, tempCP, tempCC, index, nColonies, D, imperialistPos, coloniesPos, coloniesCost) private(i)
        for (i = index; i < nE; i++) {
            if ((tempIP[i] = (double *) calloc(D, sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (removeEmpire->*tempIP[%d])\n", i);
                exit(-1);
            }
            memmove(tempIP[i], (*imperialistPos)[i + 1], D * sizeof (double));

            if ((tempCP[i] = (double *) calloc((*nColonies)[i + 1] * D, sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (removeEmpire->*tempCP[%d])\n", i);
                exit(-1);
            }
            memmove(tempCP[i], (*coloniesPos)[i + 1], (*nColonies)[i + 1] * D * sizeof (double));

            if ((tempCC[i] = (double *) calloc((*nColonies)[i + 1], sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (removeEmpire->*tempCC[%d])\n", i);
                exit(-1);
            }
            memmove(tempCC[i], (*coloniesCost)[i + 1], (*nColonies)[i + 1] * sizeof (double));
        }

        /* realloc the former pointers */
        (*imperialistCosts) = (double *) realloc(*imperialistCosts, nE * sizeof (double));
        (*totalCosts) = (double *) realloc(*totalCosts, nE * sizeof (double));
        (*nColonies) = (int *) realloc(*nColonies, nE * sizeof (int));
        (*imperialistPos) = (double **) realloc(*imperialistPos, nE * sizeof (double *));
        (*coloniesPos) = (double **) realloc(*coloniesPos, nE * sizeof (double *));
        (*coloniesCost) = (double **) realloc(*coloniesCost, nE * sizeof (double *));

        /* store the shrinked pointers within the former ones */
        memmove(*imperialistCosts, tempIC, nE * sizeof (double));
        memmove(*totalCosts, tempTC, nE * sizeof (double));
        memmove(*nColonies, tempNC, nE * sizeof (int));
        memmove(*imperialistPos, tempIP, nE * sizeof (double *));
        memmove(*coloniesPos, tempCP, nE * sizeof (double *));
        memmove(*coloniesCost, tempCC, nE * sizeof (double *));

        /* free memory */
        free(tempIC);
        free(tempTC);
        free(tempNC);
        free(tempIP);
        free(tempCP);
        free(tempCC);
    }
}

/**
 * Uniting Similar Empires
 * 
 * @param D
 * @param NI
 * @param lb
 * @param ub
 * @param uThresh
 * @param zeta
 * @param imperialistCosts
 * @param totalCosts
 * @param nColonies
 * @param imperialistPos
 * @param coloniesPos
 * @param coloniesCost
 */
void uniteSimilarEmpires(int D, int *NI, double *lb, double *ub, double uThresh, double zeta, double **imperialistCosts, double **totalCosts, int **nColonies, double ***imperialistPos, double ***coloniesPos, double ***coloniesCost) {
    int i, j, k, betterEmpIndx, worseEmpIndx, nColB, nColW, nUnCol, s, t;
    double threshDist, dist, *tempCost, *tempPos, scale;
    
#pragma omp parallel for shared(nColonies, coloniesPos, coloniesCost, totalCosts, NI, D, lb, ub, uThresh, imperialistPos, imperialistCosts) private(i, j, k, betterEmpIndx, worseEmpIndx, nColB, nColW, nUnCol, s, t, scale, threshDist, dist, tempCost, tempPos)
    for (i = 0; i < *NI - 1; i++) {
        for (j = i + 1; j < *NI; j++) {
            threshDist = 0;
            dist = 0;
            for (k = 0; k < D; k++) {
                scale = fabs(ub[k] - lb[k]);
                threshDist += pow(scale, 2);
                dist += pow((*imperialistPos)[i][k] - (*imperialistPos)[j][k], 2);
            }
            dist = pow(dist, 0.5);
            threshDist = uThresh * sqrt(threshDist);

            if (dist < threshDist) {
                betterEmpIndx = ((*imperialistCosts)[i] < (*imperialistCosts)[j]) ? i : j;
                worseEmpIndx = ((*imperialistCosts)[i] < (*imperialistCosts)[j]) ? j : i;

                /* 
                 * merge colonies and costs of actEmpires[betterEmpIndx] with
                 * colonies and costs, even imperialist, of actEmpires[worseEmpIndx]
                 */
                nColB = (*nColonies)[betterEmpIndx];
                nColW = (*nColonies)[worseEmpIndx];
                nUnCol = nColB + nColW + 1;
                
                // double tempPos[(nUnCol * D)], tempCost[nUnCol];
                if ((tempCost = (double *) calloc(nUnCol, sizeof (double))) == NULL) {
                    printf("ERROR!!! Not enough memory (uniteSimilarEmpires->*tempCost)\n");
                    exit(-1);
                }
                if ((tempPos = (double *) calloc(nUnCol * D, sizeof (double))) == NULL) {
                    printf("ERROR!!! Not enough memory (uniteSimilarEmpires->*tempPos)\n");
                    exit(-1);
                }
                
                // getting colonies of better empire
                for (s = 0; s < nColB; s++) {
                    for (t = 0; t < D; t++) {
                        tempPos[(s * D) + t] = (*coloniesPos)[betterEmpIndx][(s * D) + t];
                    }
                    tempCost[s] = (*coloniesCost)[betterEmpIndx][s];
                }
                // getting imperialist position and cost of worse empire
                for (s = 0; s < D; s++) {
                    tempPos[(nColB * D) + s] = (*imperialistPos)[worseEmpIndx][s];
                }
                tempCost[nColB] = (*imperialistCosts)[worseEmpIndx];
                // getting colonies position and cost of worse empire
                for (s = 0; s < nColW; s++) {
                    for (t = 0; t < D; t++) {
                        tempPos[((s + nColB + 1) * D) + t] = (*coloniesPos)[worseEmpIndx][(s * D) + t];
                    }
                    tempCost[(s + nColB + 1)] = (*coloniesCost)[worseEmpIndx][s];
                }
                // assign the temp arrays to the empire[betterEmpIndx]
                (*coloniesPos)[betterEmpIndx] = (double *) realloc((*coloniesPos)[betterEmpIndx], (nUnCol * D) * sizeof (double));
                (*coloniesCost)[betterEmpIndx] = (double *) realloc((*coloniesCost)[betterEmpIndx], nUnCol * sizeof (double));
                memmove((*coloniesPos)[betterEmpIndx], tempPos, (nUnCol * D) * sizeof (double));
                memmove((*coloniesCost)[betterEmpIndx], tempCost, nUnCol * sizeof (double));

                (*totalCosts)[betterEmpIndx] = (*imperialistCosts)[betterEmpIndx] +
                        (zeta * meanArray((*coloniesCost)[betterEmpIndx], nColB));
                (*nColonies)[betterEmpIndx] = nUnCol;

                /* erase the worse empire */
                removeEmpire(imperialistCosts, totalCosts, nColonies, imperialistPos, coloniesPos, coloniesCost, worseEmpIndx, NI, D);
                
                free(tempPos);
                free(tempCost);
            }
        }
    }
}

/**
 * 
 * @param prob
 * @param nEmp
 * @return 
 */
int selectAnEmpire(double *prob, int nEmp) {
    int i;
    double temp = 0;
    int indx = 0;

    srand(time(NULL));
#pragma omp parallel for shared(indx, temp, nEmp, prob) private(i)
    for (i = 0; i < nEmp; i++) {
        double act = prob[i] - ((double) rand() / ((double) (RAND_MAX) + 1.));
        if (act > temp) {
            temp = act;
            indx = i;
        }
    }

    return indx;
}

/**
 * Imperialistic Competition
 * 
 * @param D
 * @param nEmp
 * @param imperialistCosts
 * @param totalCosts
 * @param nColonies
 * @param imperialistPos
 * @param coloniesPos
 * @param coloniesCost
 */
void imperialisticCompetition(int D, int *nEmp, double **imperialistCosts, double **totalCosts, int **nColonies, double ***imperialistPos, double ***coloniesPos, double ***coloniesCost) {
    int i, j, weakestEmpIndx;
    double maxTotCosts, r;
    bool hasSingleColony;
    int dummy = *nEmp;

    /* check if competition is runnable */
    srand(time(NULL));
    r = (double) rand() / ((double) (RAND_MAX) + 1.);
    if ((r > 1) || (dummy <= 1)) { // > .11
        return;
    }
    double possessProb[dummy];

    /*
     * calculate powers and possession abilities,
     * in order to select the probabilities of possession
     */
    maxTotCosts = maxArray(*totalCosts, dummy, &weakestEmpIndx);
    int sumPowers = 0;
#pragma omp parallel for shared(dummy, maxTotCosts, totalCosts) private(i) reduction(+:sumPowers)
    for (i = 0; i < dummy; i++) {
        sumPowers += (maxTotCosts - (*totalCosts)[i]);
    }
#pragma omp parallel for shared(dummy, possessProb, maxTotCosts, totalCosts, sumPowers) private(i)
    for (i = 0; i < dummy; i++) {
        possessProb[i] = (maxTotCosts - (*totalCosts)[i]) / sumPowers;
    }

    /* randomly select and empire */
    int selEmpIndx = selectAnEmpire(possessProb, dummy);

    /* get a random colony from weakest Empire and assign it to the selected Empire */
    int nn = (*nColonies)[weakestEmpIndx];
    int jj = ceil((nn - 1) * ((double) rand() / ((double) (RAND_MAX) + 1.)));

    /* 
     * merge selected jj-th colony into selected Empire, by building temporary arrays
     * plus collapsing weakest Empire into selected one, if the former has a single
     * colony
     */
    hasSingleColony = (nn == 1) ? true : false;
    int mm = (*nColonies)[selEmpIndx];
    int tt = mm + 1;
    if (hasSingleColony == true) {
        tt += 1;
    }
    /* pour all the colonies of selEmpIndx into the temp arrays */
    double tempPosSE[tt * D], tempCostSE[tt];
#pragma omp parallel for shared(tempPosSE, tempCostSE, mm, D, coloniesPos, selEmpIndx, weakestEmpIndx, jj) private(i, j)
    for (i = 0; i < mm; i++) {
        for (j = 0; j < D; j++) {
            tempPosSE[(i * D) + j] = (*coloniesPos)[selEmpIndx][(i * D) + j];
        }
        tempCostSE[i] = (*coloniesPos)[selEmpIndx][i];
    }
    /* pour the jj-th colony of weakestEmpIndx into the temp arrays */
    for (j = 0; j < D; j++) {
        tempPosSE[(mm * D) + j] = (*coloniesPos)[weakestEmpIndx][(jj * D) + j];
    }
    tempCostSE[mm] = (*coloniesPos)[weakestEmpIndx][jj];
    /* eventually pour the weakestEmpIndx into the temp arrays */
    if (hasSingleColony == true) {
        for (j = 0; j < D; j++) {
            tempPosSE[((mm + 1) * D) + j] = (*imperialistPos)[weakestEmpIndx][j];
        }
        tempCostSE[(mm + 1)] = (*imperialistCosts)[weakestEmpIndx];
    }

    /* assign temp arrays to empire's colonies and costs */
    (*coloniesPos)[selEmpIndx] = (double *) realloc((*coloniesPos)[selEmpIndx], (tt * D) * sizeof (double));
    (*coloniesCost)[selEmpIndx] = (double *) realloc((*coloniesCost)[selEmpIndx], tt * sizeof (double));
    memmove((*coloniesPos)[selEmpIndx], tempPosSE, (tt * D) * sizeof (double));
    memmove((*coloniesCost)[selEmpIndx], tempCostSE, tt * sizeof (double));
    (*nColonies)[selEmpIndx] = tt;

    /* 
     * remove colony, from weakest Empire, by building temporary arrays
     * and collapse it if it is a colony-less Empire
     */
    int nnn = nn - 1;
    double tempPosWE[nnn * D], tempCostWE[nnn];
    // copy everything before the jj-th element
#pragma omp parallel for shared(tempPosWE, tempCostWE, jj, D, nnn, coloniesPos, weakestEmpIndx) private(i, j)
    for (i = 0; i < jj; i++) {
        for (j = 0; j < D; j++) {
            tempPosWE[(i * D) + j] = (*coloniesPos)[weakestEmpIndx][(i * D) + j];
        }
        tempCostWE[i] = (*coloniesCost)[weakestEmpIndx][i];
    }
    // copy everything after the jj-th element
#pragma omp parallel for shared(tempPosWE, tempCostWE, jj, D, nnn, coloniesPos, weakestEmpIndx) private(i, j)
    for (i = jj; i < nnn; i++) {
        for (j = 0; j < D; j++) {
            tempPosWE[(i * D) + j] = (*coloniesPos)[weakestEmpIndx][(i + 1) * D + j];
        }
        tempCostWE[i] = (*coloniesCost)[weakestEmpIndx][i + 1];
    }

    /* assign temp array to empire's colonies and costs */
    (*coloniesPos)[weakestEmpIndx] = (double *) realloc((*coloniesPos)[weakestEmpIndx], (nnn * D) * sizeof (double));
    (*coloniesCost)[weakestEmpIndx] = (double *) realloc((*coloniesCost)[weakestEmpIndx], nnn * sizeof (double));
    memmove((*coloniesPos)[weakestEmpIndx], tempPosWE, (nnn * D) * sizeof (double));
    memmove((*coloniesCost)[weakestEmpIndx], tempCostWE, nnn * sizeof (double));
    (*nColonies)[weakestEmpIndx] = nnn;

    /* erase the weakest empire, if it is colony-less */
    if (hasSingleColony == true) {
        removeEmpire(imperialistCosts, totalCosts, nColonies, imperialistPos, coloniesPos, coloniesCost, weakestEmpIndx, nEmp, D);
    }
}

/**
 * Main program of CCA (Colonial Competitive Algorithm)
 * 
 * Thanks to: "How to dynamically allocate a 2D array in C?"
 * http://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/
 * 
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char **argv) {
    printf("\t\t ---------------------------------------------\n");
    printf("\t\t|         WELCOME TO THE PROGRAM FOR          |\n");
    printf("\t\t|       COLONIAL COMPETITIVE ALGORITHM        |\n");
    printf("\t\t ---------------------------------------------\n");
    fflush(stdout);

    int nDims, nCountries, nInitImperialists, runtime, cycle, maxCycle, n, i, j,
            nAllColonies, dummy, nImperialists, *nColonies;
    double revRate, assimCoeff, zeta, dampRatio, unitingThresh, zarib, alpha,
            errorCriteria, bestCost, mean, sommaTempo, tempoMedio, std, actError,
            bestImperialistCost, *countries, *countryCosts, *bestCosts,
            *imperialistCosts, *totalCosts, **imperialistPos, **coloniesPos, **coloniesCost;
    bool stopOneEmpire;
    time_t start, stop;
    FILE *fd_ini, *fd_results_empires, *fd_results_global, *fd_results_minempires;
    char *file_results_empires, *file_results_global, *file_results_minempires;

    /* Allocate mem for filenames */
    if ((file_results_empires = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_empires)\n");
        return (-1);
    }
    if ((file_results_global = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_global)\n");
        return (-1);
    }
    if ((file_results_minempires = (char *) malloc(255 * sizeof (char))) == NULL) {
        printf("ERROR!!! Not enough memory (*file_results_minempires)\n");
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
    fscanf(fd_ini, "D=%d\n", &nDims);
    fscanf(fd_ini, "nCountries=%d\n", &nCountries);
    fscanf(fd_ini, "nInitImperialists=%d\n", &nInitImperialists);
    fscanf(fd_ini, "revRate=%lf\n", &revRate);
    fscanf(fd_ini, "assimCoeff=%lf\n", &assimCoeff);
    fscanf(fd_ini, "zeta=%lf\n", &zeta);
    fscanf(fd_ini, "dampRatio=%lf\n", &dampRatio);
    fscanf(fd_ini, "stopOneEmpire=%d\n", &stopOneEmpire);
    fscanf(fd_ini, "unitingThresh=%lf\n", &unitingThresh);
    fscanf(fd_ini, "zarib=%lf\n", &zarib);
    fscanf(fd_ini, "alpha=%lf\n", &alpha);

    double ub[nDims], lb[nDims];
    fscanf(fd_ini, "ub=%lf", &ub[0]);
    for (i = 1; i < nDims - 1; i++) {
        fscanf(fd_ini, ",%lf", &ub[i]);
    }
    fscanf(fd_ini, ",%lf\n", &ub[nDims - 1]);
    fscanf(fd_ini, "lb=%lf", &lb[0]);
    for (i = 1; i < nDims - 1; i++) {
        fscanf(fd_ini, ",%lf", &lb[i]);
    }
    fscanf(fd_ini, ",%lf\n", &lb[nDims - 1]);

    fscanf(fd_ini, "errorCriteria=%lf\n", &errorCriteria);
    fscanf(fd_ini, "runtime=%d\n", &runtime);
    fscanf(fd_ini, "maxCycle=%d\n", &maxCycle);
    fflush(fd_ini);
    fclose(fd_ini);

    nAllColonies = nCountries - nInitImperialists;
    revRate = dampRatio * revRate;
    nImperialists = nInitImperialists

    /* allocateMem */
    printf("Allocating memory\n");
    fflush(stdout);
    if ((countries = (double *) calloc(nCountries * nDims, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*countries)\n");
        return (-1);
    }
    if ((countryCosts = (double *) calloc(nCountries, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*countryCosts)\n");
        return (-1);
    }
    if ((bestCosts = (double *) calloc(maxCycle, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*bestCosts)\n");
        return (-1);
    }
    if ((imperialistCosts = (double *) calloc(nImperialists, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*imperialistCosts)\n");
        return (-1);
    }
    if ((totalCosts = (double *) calloc(nImperialists, sizeof (double))) == NULL) {
        printf("ERROR!!! Not enough memory (*totalCosts)\n");
        return (-1);
    }
    if ((nColonies = (int *) calloc(nImperialists, sizeof (int))) == NULL) {
        printf("ERROR!!! Not enough memory (*nColonies)\n");
        return (-1);
    }
    if ((imperialistPos = (double **) calloc(nImperialists, sizeof (double *))) == NULL) {
        printf("ERROR!!! Not enough memory (**imperialistPos)\n");
        return (-1);
    }
    if ((coloniesPos = (double **) calloc(nImperialists, sizeof (double *))) == NULL) {
        printf("ERROR!!! Not enough memory (**coloniesPos)\n");
        return (-1);
    }
    if ((coloniesCost = (double **) calloc(nImperialists, sizeof (double *))) == NULL) {
        printf("ERROR!!! Not enough memory (**coloniesCost)\n");
        return (-1);
    }
    // alloc memory for double pointer imperialistPos
    for (i = 0; i < nImperialists; i++) {
        if ((imperialistPos[i] = (double *) calloc(nDims, sizeof (double))) == NULL) {
            printf("ERROR!!! Not enough memory (imperialistPos[%d])\n", i);
            return (-1);
        }
    }

    /* open file results */
    time_t now = time(NULL);
    struct tm *ptr;
    ptr = localtime(&now);
    strftime(file_results_empires, 45, "results_empires_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_global, 45, "results_global_%Y%m%d%H%M%S.dat", ptr);
    strftime(file_results_minempires, 45, "results_minempires_%Y%m%d%H%M%S.dat", ptr);

    /* open file for writing results */
    fd_results_empires = fopen(file_results_empires, "wb+");
    fd_results_global = fopen(file_results_global, "wb+");
    fd_results_minempires = fopen(file_results_minempires, "wb+");

    /* write headers into files */
    // results for empires
    fprintf(fd_results_empires, "#Runtime,#Cycle,#Empire");
    for (i = 0; i < nDims; i++) {
        fprintf(fd_results_empires, ",Dim[%d]", i);
    }
    fprintf(fd_results_empires, "\n");
    // results for minempires
    fprintf(fd_results_minempires, "#Runtime,#Cycle");
    for (i = 0; i < nDims; i++) {
        fprintf(fd_results_minempires, ",Dim[%d]", i);
    }
    fprintf(fd_results_minempires, ",ImperialistCost,TotalCost,NumColonies,Error\n");

    /* runs for optimization; algorithm can be run multiple times in order to check its robustness */
    sommaTempo = 0;
    mean = 0;
    for (n = 0; n < runtime; n++) {
        time(&start);
        printf("Starting optimization run #%d\n", n + 1);
        fflush(stdout);

        /* initialize countries and empires */
        printf("\tInitializing system\n");
        fflush(stdout);
        
        // init countries
        initCountries(nCountries, nDims, lb, ub, countries, countryCosts);

        // init characteristics of empires
        initEmpires(nImperialists, nDims, nAllColonies, countries, countryCosts, imperialistCosts, imperialistPos, nColonies);
        // alloc memory for double pointers coloniesPos and coloniesCost, depending on nColonies
        for (i = 0; i < nImperialists; i++) {
            int indHigh = nColonies[i];
            if ((coloniesPos[i] = (double *) calloc(indHigh * nDims, sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (coloniesPos[%d])\n", i);
                exit(-1);
            }
            if ((coloniesCost[i] = (double *) calloc(indHigh, sizeof (double))) == NULL) {
                printf("ERROR!!! Not enough memory (coloniesCost[%d])\n", i);
                exit(-1);
            }
        }
        // init colonies of empires
        initColoniesOfEmpires(nImperialists, nCountries, nDims, nAllColonies, zeta, lb, ub, countries, countryCosts, imperialistCosts, totalCosts, nColonies, coloniesPos, coloniesCost);

        /* optimization cycles */
        printf("\tStarting cycles");
        fflush(stdout);
        cycle = 1;
        do {
            printf(" #%d", cycle);
            fflush(stdout);
            
            int nIters = nImperialists;
            for (i = 0; i < nIters; i++) {
                /* Assimilation; Movement of Colonies Toward Imperialists (Assimilation Policy) */
                assimilateColonies(nDims, nColonies[i], imperialistPos[i], coloniesPos[i], lb, ub, assimCoeff);

                /* Revolution;  A Sudden Change in the Socio-Political Characteristics */
                revolveColonies(nDims, revRate, lb, ub, nColonies[i], coloniesPos[i]);

                /* New Cost Evaluation */
                evalNewColoniesPosition(nDims, nColonies[i], coloniesPos[i], coloniesCost[i]);

                /* Empire Possession (++++++ Power Possession, Empire Possession) */
                possessEmpire(nDims, imperialistCosts[i], nColonies[i], imperialistPos[i], coloniesPos[i], coloniesCost[i]);

                /* Computation of Total Cost for Empires */
                totalCosts[i] = imperialistCosts[i] + zeta * meanArray(coloniesCost[i], nColonies[i]);
            }
            /* Uniting Similar Empires */
            uniteSimilarEmpires(nDims, &nImperialists, lb, ub, unitingThresh, zeta, &imperialistCosts, &totalCosts, &nColonies, &imperialistPos, &coloniesPos, &coloniesCost);

            /* Imperialistic Competition */
            imperialisticCompetition(nDims, &nImperialists, &imperialistCosts, &totalCosts, &nColonies, &imperialistPos, &coloniesPos, &coloniesCost);            

            bestImperialistCost = minArray(imperialistCosts, nImperialists, &dummy);
            actError = CalculateFitness(bestImperialistCost);

            /* saving results into files */
            // empires
            for (i = 0; i < nImperialists; i++) {
                fprintf(fd_results_empires, "%d,%d,%d", n + 1, cycle, i + 1);
                for (j = 0; j < nDims; j++) {
                    fprintf(fd_results_empires, ",%4.3f", imperialistPos[i][j]);
                }
                fprintf(fd_results_empires, "\n");
            }
            // minempires
            fprintf(fd_results_minempires, "%d,%d", n + 1, cycle);
            for (i = 0; i < nDims; i++) {
                fprintf(fd_results_minempires, ",%4.3f", imperialistPos[0][i]);
            }
            fprintf(fd_results_minempires, ",%4.3f,%4.3f,%d,%4.3f\n", imperialistCosts[0], totalCosts[0], nColonies[0], actError);

            cycle++;

            if ((nImperialists == 1) && (stopOneEmpire == true)) {
                break;
            }
        } while ((cycle <= maxCycle) && (actError > errorCriteria));

        bestCost = minArray(imperialistCosts, nImperialists, &dummy);
        bestCosts[n] = bestCost;
        mean += bestCost;

        time(&stop);
        printf("\n\tElapsed time: %.3f s - BestCost: %4.3f\n", difftime(stop, start), bestCost);
        fflush(stdout);

        /* save info into global files */
        fprintf(fd_results_global, "Runtime #%d, elapsed time %.3f s, BestCost: %4.3f\n",
                n + 1, difftime(stop, start), bestCost);

        sommaTempo += difftime(stop, start);
        
        free(coloniesPos);
        free(coloniesCost);
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

    fflush(fd_results_empires);
    fclose(fd_results_empires);
    fflush(fd_results_minempires);
    fclose(fd_results_minempires);
    fflush(fd_results_global);
    fclose(fd_results_global);

    /* deallocate mem */
    free(file_results_empires);
    free(file_results_minempires);
    free(file_results_global);
    free(countries);
    free(countryCosts);
    free(bestCosts);
    free(imperialistCosts);
    free(totalCosts);
    free(nColonies);
    free(imperialistPos);

    return (0);
}