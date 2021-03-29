#include <stdio.h>
#include <stdlib.h>
#include "SNNC 2.0.h"

///***************Predicting forex with linear regression***************\\\

///Hyper parameters
int l1_size = 6;
int l2_size = 2;
int l3_size = 1;
int data_size = 2645;
int epochs = 9000;

///learning rate
double lr = 0.000001;

///Layers
Layer l1, l2, l3;


///Vectors with data
double datas[3000][7];
double answers[3000][1];

int main()
{
    printf("Predicting forex close value for today\n\n");
    ///data input
    printf("dataset_size:"); //(input the dataset number of lines)
    scanf("%d", &data_size);

    printf("learning_rate:"); //(input the learning rate)
    scanf("%lf", &lr); //if the values generated while training are infinite make the learning rate smaller

    printf("epochs:"); //(input the epochs)
    scanf("%d", &epochs);
    printf("\n");

    ///initializing layers
    Init_Input_Layer(&l1, &l3, l1_size, l3_size,Linear);
    Init_Output_Layer(&l3, &l1, l3_size, Linear);

    ///data loading
    FILE *fp;
    fp = fopen("dataset_C.in", "r"); //select dataset_L.in for predicting the lows, dataset_H.in for predicting the Highs and dataset_C.in for predicting closing price
    for(int i=0; i<data_size; i++){
        for(int j=0; j<=l1_size; j++)
        fscanf(fp, "%lf", &datas[i][j]);
    }
    fclose(fp);

    ///setting answers
    for(int i=0; i<data_size; i++)
        answers[i][0]=datas[i][6];

    ///training
    for(int iter=0; iter<=epochs*2; iter++){
        for(int i=1000; i<data_size-1; i++)
        {
            for(int j=0; j<l1_size; j++)
                l1.neuron[j].Value = datas[i][j];
            FeedForward_L(&l3);
            if(iter%1000==0 && i>data_size-50)
                printf("error: %f, %f, %f\n", Calc_SqError(&l3, answers[i]), l3.neuron[0].Value, answers[i][0]);
            BackProp_O(&l3, answers[i], lr);
        }
        if(iter%1000==0)
        printf("\n");
    }

    ///show prediction (prediction is not very accurate)

    for(int i=0; i<l1_size; i++)
    {
        l1.neuron[i].Value = datas[data_size-1][i];
        printf("%f\n", l1.neuron[i].Value);
    }
    printf("\nFinished training\nShowing results\n\n");
    FeedForward_L(&l3);
    printf("Pediction: %f\n\n", l3.neuron[0].Value);
    while(true);
    return 0;
}

