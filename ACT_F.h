#ifndef ACT_F_H_INCLUDED
#define ACT_F_H_INCLUDED
#include <math.h>

///*********ACTIVATION FUNCTIONS*********\\\

double sigmoid(double x)
{
    return 1/(1+exp(-x));
}

double sigmoid_p(double x)
{
    return x*(1-x);
}

double TanH(double x)
{
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

double TanH_p(double x)
{
    return 1-pow(x, 2);
}

double ReLU(double x)
{
    if(x>0)
        return x;
    return 0;
}

double ReLU_p(double x)
{
    if(x>0)
        return 1;
    return 0;
}

double Linear(double x)
{
    return x;
}

double Linear_p(double x)
{
    return 1;
}

double LReLU(double x)
{
    if(x>0)
        return x;
    return 0.07*x;
}

double LReLU_p(double x)
{
    if(x>0)
        return 1;
    return 0.07;
}

double one(double x)
{
    return 1;
}
#endif // ACT_F_H_INCLUDED
