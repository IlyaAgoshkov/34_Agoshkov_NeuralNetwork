using System;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using static System.Math;

namespace _34_Agoshkov_NeuralNetwork.ModelNeuroNet
{
    abstract class Layer
    {
        protected string name_Layer;
        string pathDirWeights;
        string pathFileWeights;
        protected int numofneurons;
        protected int numofprevneurons;
        protected const double learningrate = 0.1d;
        protected const double momentum = 0.01d;
        protected double[,] lastdeltaweights;
        Neuron[] _neurons;

        public Neuron[] Neurons
        {
            get { return _neurons; }
            set { _neurons = value; }
        }
        public double[] Data
        {
            set
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i].Input = value;
                    Neurons[i].Activator();
                }
            }
        }
        protected Layer(int non, int nopn, TypeNeuron tn, string nm_Layer)
        {
            numofneurons = non;
            numofprevneurons= nopn;
            Neurons=new Neuron[non];
            name_Layer=nm_Layer;
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";
            
            double[,] Weights;
            if (File.Exists(pathFileWeights))
                Weights = WeightInitialize(MemoryMode.Get, pathFileWeights);
            else { 
                Directory.CreateDirectory(pathDirWeights);
                Weights=WeightInitialize(MemoryMode.Init,pathFileWeights);
            }

            lastdeltaweights = new double[non, nopn + 1];

            for(int i = 0; i < non; i++)
            {
                double[] tmp_weights = new double[nopn+1];
                for(int j=0; j<nopn+1; j++)
                {
                    tmp_weights[j] = Weights[i,j];

                }
                Neurons[i] = new Neuron(null, tmp_weights, tn);
            }
        }


       public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' };
            string tmpStr;
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            switch (mm)
            {

                case MemoryMode.Get:
                    tmpStrWeights = File.ReadAllLines(pathFileWeights);
                    string[] memory_element;
                    for (int i = 0; i < numofneurons; i++)
                    {

                        Console.WriteLine(tmpStrWeights.Length);
                        memory_element = tmpStrWeights[i].Split(delim[0]);
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_element[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;
                case MemoryMode.Set:
                    tmpStrWeights = new string[numofneurons];
                    if (!File.Exists(path))
                    {
                        MessageBox.Show("Внимание, \nФайл синаптических весов отсутсвует, после нажатия Ок, он создатся автоматически");
                    }
                    for(int i=0; i < numofneurons; i++)
                    {
                        tmpStr = Neurons[i].Weight[0].ToString();
                        for(int j=1; j < numofprevneurons+1; j++)
                        {
                            tmpStr += delim[0] + Neurons[i].Weight[j].ToString();
                        }
                        tmpStrWeights[i] = tmpStr;
                    }
                    File.WriteAllLines(path,tmpStrWeights); 
                    break;
                case MemoryMode.Init:
                    /*
               1. инициальзация весов случайными величинами 
               2. среднее значение всех весов нейрона должно равняться 0. Найти среднее значение и вычесть из вех знчений
               3. среднее квадратич знач должно равняться 1
                           */
                    Random random = new Random();
                    double[] tmpArr = new double[numofprevneurons + 1];
                    tmpStrWeights = new string[numofneurons];



                    tmpStr = "";
                    for (int i = 0; i < numofneurons; i++)
                    {
                        for (int j = 0; j < numofprevneurons + 1; j++)
                            tmpArr[j] = 0.02 * random.NextDouble() - 0.01;

                        double tmpRatio = 1.0d / Math.Sqrt(Calc_Dispers(tmpArr) * (numofprevneurons + 1));
                        double tmpShift = Calc_Average(tmpArr);
                        weights[i, 0] = (tmpArr[0] - tmpShift) * tmpRatio;
                        tmpStr = weights[i, 0].ToString();


                        for (int j = 1; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = (tmpArr[j] - tmpShift) * tmpRatio;
                            tmpStr += delim[0] + weights[i, j].ToString();
                        }
                        tmpStrWeights[i] = tmpStr;
                    }
                    File.WriteAllLines(path, tmpStrWeights);
                    break;
                default:
                    break;
            }
            return weights;


        }


        protected double Calc_Average(double[] arr)
        {

            return arr.Sum() / arr.Length;
        }

        protected double Calc_Dispers(double[] arr)
        {
            double mean = Calc_Average(arr);

            double[] squaredDifferences = new double[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                //Вычесть среднее значение из каждого элемента. Получившиеся разности возводят в квадрат.
                squaredDifferences[i] = Math.Pow(arr[i] - mean, 2);
            }

            double dispersion = squaredDifferences.Sum() / squaredDifferences.Length;
            //Вычислить среднее значение квадратов разностей. Для этого суммируются все квадраты разностей, а затем полученная сумма делится на количество чисел в массиве. Это и есть дисперсия.
            return dispersion;
        }
        abstract public void Recognize(NeuroNet net, Layer nextLayer);
        abstract public double[] BackwardPass(double[] stuff);


    }
}










/*
                    1. инициальзация весов случайными величинами 
                    2. среднее значение всех весов нейрона должно равняться 0. Найти среднее значение и вычесть из вех знчений
                    3. среднее квадратич знач должно равняться 1
                                */