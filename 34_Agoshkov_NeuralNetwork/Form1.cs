using _34_Agoshkov_NeuralNetwork.ModelNeuroNet;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace _34_Agoshkov_NeuralNetwork
{
    public partial class Form_Main : Form
    {

        private double[] inputData = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        public Chart ChartControl
        {
            get { return chart1; }
        }

        private NeuroNet net;
        public Form_Main()
        {
            InitializeComponent();
            net = new NeuroNet(NetworkMode.Rec, this);
        }

 

        void ChangeColor(Button b, int index)
        {
            if (b.BackColor == Color.Aquamarine)
            {
                b.BackColor = Color.Black;
                inputData[index] = 1;
               
            }
            else
            {
                b.BackColor = Color.Aquamarine;
                inputData[index] = 0;
            }
            label1.Text = string.Join("", inputData);
        }


        private void button1_Click(object sender, EventArgs e) => ChangeColor(button1, 0);
        private void button2_Click(object sender, EventArgs e) => ChangeColor(button2, 1);
        private void button3_Click(object sender, EventArgs e) => ChangeColor(button3, 2);

        private void button4_Click(object sender, EventArgs e) => ChangeColor(button4, 3);
        private void button5_Click(object sender, EventArgs e) => ChangeColor(button5, 4);
        private void button6_Click(object sender, EventArgs e) => ChangeColor(button6, 5);
        private void button7_Click(object sender, EventArgs e) => ChangeColor(button7, 6);
        private void button8_Click(object sender, EventArgs e) => ChangeColor(button8, 7);
        private void button9_Click(object sender, EventArgs e) => ChangeColor(button9, 8);
        private void button10_Click(object sender, EventArgs e) => ChangeColor(button10, 9);
        private void button11_Click(object sender, EventArgs e) => ChangeColor(button11, 10);
        private void button12_Click(object sender, EventArgs e) => ChangeColor(button12, 11);
        private void button13_Click(object sender, EventArgs e) => ChangeColor(button13, 12);
        private void button14_Click(object sender, EventArgs e) => ChangeColor(button14, 13);
        private void button15_Click(object sender, EventArgs e) => ChangeColor(button15, 14);
        private void button_test_Click(object sender, EventArgs e)
        {

        }

        private void button_rec_Click(object sender, EventArgs e)
        {
            net.ForwardPass(net, inputData);
        
            label2.Text = net.fact.ToList().IndexOf(net.fact.Max()).ToString();
        }

        private void button_save_Click(object sender, EventArgs e)
        {
            string pathFileTrainSample = AppDomain.CurrentDomain.BaseDirectory + "trainSample.txt";
            string strSample = numericUpDown1.Value.ToString();
            for (int i = 0; i < inputData.Length; i++)
            {
                strSample += " " + inputData[i].ToString();
            }
            strSample += "\n";

            File.AppendAllText(pathFileTrainSample, strSample);
        }

        private void Form_Main_Load(object sender, EventArgs e)
        {

        }

        private void button_learn_Click(object sender, EventArgs e)
        {

            net.Train(net);
            
        }

        private void chart1_Click(object sender, EventArgs e)
        {

        }
    }
}
