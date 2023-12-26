using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;
namespace _34_Agoshkov_NeuralNetwork.ModelNeuroNet
{
    class Neuron
    {
        TypeNeuron type;
        double[] weight;
        double[] inputData;
        double outputData;
        private double _derivative;

        public double []Weight
        {
            get { return weight; }
            set { weight = value; }
        }

        public double[] Input
        {
            get { return inputData; }
            set { inputData = value; } 
        }
        public double Output { get { return outputData; }}
        public double Derivative { get => _derivative; }

        public Neuron(double[] inpt,  double[]w, TypeNeuron typeNeuron)
        {
            type = typeNeuron;
            weight = w;
            inputData = inpt;
        }
        public void Activator(double[] inputD, double[] w)
        {
            double weightedSum = w[0];
            for (int i = 0; i < inputD.Length; i++)
            {
                weightedSum += inputD[i] * w[i + 1];

            }

            outputData = 1 / (1 + Math.Exp(weightedSum));
        }
        public void Activator()
        {
            double sum;
            sum = weight[0];
            for (int i = 0; i < inputData.Length; i++)
            {
                sum += inputData[i] * weight[i + 1];
            }
            switch (type)
            {
                case TypeNeuron.HiddenLayer:
                    outputData = Logistic(sum);
                    _derivative = Logistic_Derivative(sum);
                    break;
                case TypeNeuron.OutputLayer:
                    outputData = Exp(sum);
                    break;
            }
        }

        private double Logistic(double arg)
        {
            return 1 / (1 + Math.Exp(-arg));
        }

        private double Logistic_Derivative(double arg)
        {
            return Math.Exp(-arg) / ((1 + Math.Exp(-arg) * (1 + Math.Exp(-arg))));
        }

    }
}
