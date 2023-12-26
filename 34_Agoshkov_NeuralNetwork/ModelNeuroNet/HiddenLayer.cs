using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace _34_Agoshkov_NeuralNetwork.ModelNeuroNet
{
    class HiddenLayer : Layer
    {
public HiddenLayer(int non, int nopn, TypeNeuron nt, string name) : base(non, nopn, nt, name) { }
        
        public override void Recognize(NeuroNet net, Layer nextLayer)
        {
            double[] hidden_out = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
                hidden_out[i] = Neurons[i].Output;
            nextLayer.Data= hidden_out;
        }
        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];
            for (int j = 0; j < gr_sum.Length; j++) // Вычисление градиентных сумм
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                    sum += Neurons[k].Weight[j] * Neurons[k].Derivative * gr_sums[k];
            }

            for (int i = 0; i < numofneurons; i++) // Вычисление коррекции синапт. весов
            {
                for (int n = 0; n < numofprevneurons; n++)
                {
                    double deltaW = 0;
                    if (n == 0) // Для коррекции порогов
                        deltaW = momentum * lastdeltaweights[i, 0] + learningrate * Neurons[i].Derivative * gr_sums[i];
                    else 
                        deltaW = momentum * lastdeltaweights[i, n] + 
                            learningrate * Neurons[i].Input[n - 1] * Neurons[i].Derivative * gr_sums[i];
                    lastdeltaweights[i, n] = deltaW;
                    Neurons[i].Weight[n] += deltaW; // Коррекция весов
                }
            }
            return gr_sum;
        }
    }
}
