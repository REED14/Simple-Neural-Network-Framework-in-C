#ifndef SNNC_2_0_H_INCLUDED
#define SNNC_2_0_H_INCLUDED
#include "ACT_F.h"
#include <stdbool.h>

int seed = 101;
bool activated = 0;

typedef struct Layer Layer;

///the structure of a neuron
typedef struct Neuron
{
    double *weights;
    double bias;
    double Value;
    double error;
    double (*activation)(double);

    int Next_LS;
    Layer* next_layer;
    Layer* prev_layer;
}Neuron;

///the structure of a layer (has an array of neurons)
struct Layer{
    int Layer_Size;
    int Next_LS;
    Neuron *neuron;
    Layer* prev_layer;
    Layer* next_layer;
};

///generates weights for every neuron
void generate_weights(Neuron* n)
{
    if(!activated)
    {srand(seed); activated=1;}

    for(int i=0; i<n->Next_LS; i++){
        n->weights[i]=(float)rand()/(float)(RAND_MAX)-0.5; ///generates values between -0.5 and 0.5
    }
}

///initializes (hidden layer) neuron
void Init_Neuron(Neuron *n,Layer* prev_layer, Layer* next_layer, int Next_LS, double (*activ)(double))
{
    n->next_layer = next_layer;
    n->Next_LS = Next_LS;
    n->weights = (double*) calloc(Next_LS, sizeof(double));
    generate_weights(n);
    n->bias = 1;
    n->activation = activ;
    n->prev_layer = prev_layer;
    n->Value = 0;
}

///initializes input neuron
void Init_Input_Neuron(Neuron *n, Layer* next_layer,int Next_LS, double (*activ)(double))
{
    n->next_layer = next_layer;
    n->Next_LS = Next_LS;
    n->weights = (double*) calloc(Next_LS, sizeof(double));
    generate_weights(n);
    n->bias = 1;
    n->activation = activ;
    n->Value = 0;
}

///initializes output neuron
void Init_Output_Neuron(Neuron *n, Layer* prev_layer, double (*activ)(double))
{
    n->bias = 0;
    n->activation = activ;
    n->prev_layer = prev_layer;
    n->Value = 0;
}

///sets the value of the neuron
void SetNeuronValue(Neuron *n, double x)
{
    n->Value = x;
}

///sets the value of the neuron to 0 (clears the value)
void ClearNeuronValue(Neuron* n)
{
    n->Value=0;
}

///calculates the value of the neuron using the previous layer
void FeedForward_N(Neuron* n, Layer* prev_layer, int pos_n_in_layer)
{
    ClearNeuronValue(n);
    for(int i=0; i<prev_layer->Layer_Size; i++)
        n->Value += prev_layer->neuron[i].weights[pos_n_in_layer] * prev_layer->neuron[i].Value;
    n->Value += n->bias;
    n->Value = n->activation(n->Value);
}

///initialize (hidden) layer
void Init_Layer(Layer* l,Layer* l_prev, Layer* l_next, int LS, int Next_LS, double (*activ)(double))
{
    l->prev_layer = l_prev;
    l->next_layer = l_next;
    l->Layer_Size = LS;
    l->Next_LS = Next_LS;
    l->neuron = (Neuron*) malloc(LS* sizeof(Neuron));
    for(int i=0; i<LS; i++)
    {
        Init_Neuron(l->neuron+i, l_prev, l_next, Next_LS, activ);
    }
}

///initialize layer of input neurons
void Init_Input_Layer(Layer* l, Layer* l_next, int LS, int Next_LS, double (*activ)(double))
{
    l->next_layer = l_next;
    l->Layer_Size = LS;
    l->Next_LS = Next_LS;
    l->neuron = (Neuron*) malloc(LS* sizeof(Neuron));
    for(int i=0; i<LS; i++)
    {
        Init_Input_Neuron(l->neuron+i, l_next, Next_LS, activ);
    }
}

///initialize layer of output neurons
void Init_Output_Layer(Layer* l,Layer* l_prev, int LS, double (*activ)(double))
{
    l->prev_layer = l_prev;
    l->Layer_Size = LS;
    l->neuron = (Neuron*) malloc(LS* sizeof(Neuron));
    for(int i=0; i<LS; i++)
    {
        Init_Output_Neuron(l->neuron+i, l_prev, activ);
    }
}

///sets the value of every neuron in layer to zero (clears the values)
void ClearLayerValues(Layer *l)
{
    for(int i=0; i<l->Layer_Size; i++)
        l->neuron[i].Value=0;
}

///calculates the value of the neurons of the current using the previous layer
void FeedForward_L(Layer *l)
{
    for(int i=0; i<l->Layer_Size; i++)
        FeedForward_N(l->neuron+i, l->prev_layer, i);
}

///calculates the squared errors of output
double Calc_SqError(Layer *output ,double *answers)
{
    double res = 0;
    for(int i=0; i<output->Layer_Size; i++)
        res +=(output->neuron[i].Value-answers[i])*(output->neuron[i].Value-answers[i]);
    return res;
}

///backpropagate the error from output to previous layer ( linear/logistic regression)
void BackProp_O(Layer *output, double answers[], double lr)
{
    double error, prime,dW;
    for(int i=0; i<output->Layer_Size; i++)
    {
        //calculates the error
        error = 2*(output->neuron[i].Value-answers[i]);

        //selects the derivative of activation function and calculates the derivative of output neuron value(prime)
        if(output->neuron[i].activation == ReLU)
            prime = ReLU_p(output->neuron[i].Value);
        if(output->neuron[i].activation == sigmoid)
            prime = sigmoid_p(output->neuron[i].Value);
        if(output->neuron[i].activation == TanH)
            prime = TanH_p(output->neuron[i].Value);
        if(output->neuron[i].activation == LReLU)
            prime = LReLU_p(output->neuron[i].Value);
        if(output->neuron[i].activation == Linear)
            prime = 1;
        //multiplies the error with prime
        error *= prime;

        //saves the error to neuron
        output->neuron[i].error = error;

        //applies the error to the weights and biases
        for(int j=0; j<output->prev_layer->Layer_Size; j++)
        {
            dW = output->prev_layer->neuron[j].Value * error;
            output->prev_layer->neuron[j].weights[i] -= dW*lr;
        }
        output->neuron[i].bias -= error*lr;
    }

}

///backpropagate the error from hidden to previous layer (linear/logistic regression)
void BackProp_L(Layer *current, double lr)
{
    double error, prime, dW;
    for(int i=0; i<current->Layer_Size; i++)
    {
        //gets error from next layer
        error = 0;
        for(int j=0; j<current->next_layer->Layer_Size; j++)
        {
            error += current->next_layer->neuron[j].Value * current->neuron[i].weights[j];
        }
        //calculates the average error (normalizes the error)
        error = error / current->next_layer->Layer_Size;

        //selects the derivative of activation function and calculates the derivative of hidden neuron value(prime)
        if(current->neuron[i].activation == ReLU)
            prime = ReLU_p(current->neuron[i].Value);
        if(current->neuron[i].activation == sigmoid)
            prime = sigmoid_p(current->neuron[i].Value);
        if(current->neuron[i].activation == TanH)
            prime = TanH_p(current->neuron[i].Value);
        if(current->neuron[i].activation == LReLU)
            prime = LReLU_p(current->neuron[i].Value);
        if(current->neuron[i].activation == Linear)
            prime = 1;
        //multiplies the error with prime
        error *= prime;

        //saves the error to neuron
        current->neuron[i].error = error;

        //applies the error to the weights and biases
        for(int j=0; j<current->prev_layer->Layer_Size; j++)
        {
            dW = error * current->prev_layer->neuron[j].Value;
            current->prev_layer->neuron[j].weights[i] -= dW*lr;
        }
        current->neuron[i].bias -= error*lr;
    }
}

#endif // SNNC_2_0_H_INCLUDED
