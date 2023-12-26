using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.Windows.Forms.DataVisualization.Charting;
namespace _34_Agoshkov_NeuralNetwork.ModelNeuroNet
{
    class NeuroNet
    {
        //все слои нейросети

        InputLayer input_Layer;
        private HiddenLayer hidden_Layer1 = new HiddenLayer(70, 15, TypeNeuron.HiddenLayer, nameof(hidden_Layer1));
        private HiddenLayer hidden_Layer2 = new HiddenLayer(31, 70, TypeNeuron.HiddenLayer, nameof(hidden_Layer2));
        private OutputLayer output_Layer = new OutputLayer(10, 31, TypeNeuron.OutputLayer, nameof(output_Layer));
        public List<double> epochErrors = new List<double>();
        public Series series = new Series();
        Chart chart;
        public double[] fact = new double[10];
        //среднее значение энергии ошибки эпохи обучения
        private double e_error_avr;
        public double E_error_avr
        {
            get => e_error_avr;
            set => e_error_avr = value;
        }
        public NeuroNet(NetworkMode nm, Form_Main form)
        {
            input_Layer = new InputLayer(nm);
            chart = form.ChartControl;
            series.ChartType = SeriesChartType.Line;
            chart.Series.Add(series);
        }
        //прямой проход нейросети
        public void ForwardPass(NeuroNet net, double[] netInput)
        {
            net.hidden_Layer1.Data = netInput;
            net.hidden_Layer1.Recognize(null, net.hidden_Layer2);
            net.hidden_Layer2.Recognize(null, net.output_Layer);
            net.output_Layer.Recognize(net, null);
        }

        public async Task Train(NeuroNet net)
        {


            int epochs = 150;
            net.input_Layer = new InputLayer(NetworkMode.Train);

            double tmpSumError; // Временная переменная суммы ошибок
            double[] errors; //Вектор сигнала ошибки выходного слоя
            double[] tmpGradSums1; //Вектор градиента первого скрытого слоя
            double[] tmpGradSums2; //Вектор градиента второго скрытого слоя

            for (int k = 0; k < epochs; k++)
            {
                e_error_avr = 0; // Обнуляем значение в начале эпохи
                for (int i = 0; i < net.input_Layer.TrainSet.Length; i++)
                {
                    //Прямой проход
                    ForwardPass(net, net.input_Layer.TrainSet[i].Item1);

                    //Вычисление ошибки по итерации
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {

                        if (x == net.input_Layer.TrainSet[i].Item2)
                        {
                            errors[x] = 1.0 - net.fact[x];

                        }
                        else
                        {
                            errors[x] = -net.fact[x];

                        }

                        //Собираем энергию ошибки
                        tmpSumError += errors[x] * errors[x] / 2.0;
                    }

                    e_error_avr += tmpSumError / errors.Length; //Суммарное значение энергии оишбки эпох

                    //Обратный проход и коррекция весов
                    tmpGradSums2 = net.output_Layer.BackwardPass(errors);
                    tmpGradSums1 = net.hidden_Layer2.BackwardPass(tmpGradSums2);
                    net.hidden_Layer1.BackwardPass(tmpGradSums1);
                }
                Console.WriteLine($"Эпоха {k}, ошибка: {e_error_avr}");
                epochErrors.Add(e_error_avr);


                series.Points.AddY(e_error_avr);
                chart.Update();

                e_error_avr /= net.input_Layer.TrainSet.Length; //Среднее значение энергии ошибки одной эпохи


            }

            net.input_Layer = null; //Обнуление входного слоя

            //Сохранение скорректированных весов
            net.hidden_Layer1.WeightInitialize(MemoryMode.Set, "memory/" + nameof(hidden_Layer1) + "_memory.csv");
            net.hidden_Layer2.WeightInitialize(MemoryMode.Set, "memory/" + nameof(hidden_Layer2) + "_memory.csv");
            net.output_Layer.WeightInitialize(MemoryMode.Set, "memory/" + nameof(output_Layer) + "_memory.csv");

        }



    }

}
