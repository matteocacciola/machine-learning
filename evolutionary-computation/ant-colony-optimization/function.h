#include <math.h>

/* 
 * a function pointer returning double and taking a D-dimensional array as argument
 * If your function takes additional arguments then change function pointer definition and lines 
 * calling "...=function(solution);" in the code
 */
typedef double (*FunctionCallback) (double*, int);

/*
 * Write your own objective function name instead of Rastrigin
 * 
 * @param sol
 * @param D
 * @return 
 */
double Rastrigin(double *sol, int D) {
    int j;
    double top = 0;
    for (j = 0; j < D; j++) {
        top += (pow(sol[j], 2) - 10 * cos(2 * M_PI * sol[j]) + 10);
    }
    return top;
}
FunctionCallback function = &Rastrigin;