package me.hagen.nd4jtest;

import java.util.HashSet;
import java.util.Set;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import me.hagen.neural.Util;
import me.hagen.neural.fuc.Linear;
import me.hagen.neural.fuc.Sigmoid;
import me.hagen.neural.optimize.Abstract.Gradients;
import me.hagen.neural.optimize.Adagrad;
import me.hagen.neural.optimize.Adam;
import me.hagen.neural.optimize.Momentum;
import me.hagen.neural.unit.Gate;
import me.hagen.neural.optimize.SGD;

public class Nd4jTest {
	@Test
	public void testAdd() {
		INDArray data = Nd4j.create(new double[] {1.0,2.0});
		INDArray data2 = data.add(10);
		INDArray data3 = data.add(Nd4j.create(new double[] {0.1,0.2}));
		System.out.println(data);
		System.out.println(data2);
		System.out.println(data3);
		System.out.println(data.mmul(data.transpose()));
	}
	
	@Test
	public void testHamord() {
		INDArray x = Nd4j.create(new double[] {2.0,3.0});
		//x = x.transpose();
		INDArray delta = Nd4j.create(new double[] {0.1,0.3,0.5});
		System.out.println(Util.getDeltaW(delta, x));
	}
	@Test
	public void testGate2() {
		int inputSize = 1;
		int hiddenSize = 1;
		int biasLen = 1;
		Gate g = new Gate(inputSize,hiddenSize,biasLen,new Linear(),new Adam());
		INDArray w = Nd4j.create(new double[] {5});
		INDArray b = Nd4j.create(new double[] {3});
		INDArray h = Nd4j.create(new double[] {0});
		INDArray u = Nd4j.create(new double[] {0});
		INDArray x = Nd4j.rand(new int[] {1});
		INDArray y = g.forward(x, h, w, u, b);
		System.err.println("w:"+w+" b:"+b);	
		System.out.println("x:"+x+" y:"+y);
		System.out.println(g.status());
		INDArray ypredict = g.forward(x, h);
		System.out.println("y predict:"+ypredict);
		Gradients up = g.backward(x, h, ypredict.sub(y));
		g.printUpdater(up);
	}
	@Test
	public void testGate3() {
		int setSize = 1000;
		int batchSize = 50;
		int batchTime = 10000;
		int inputSize = 2;
		int hiddenSize = 2;
		int biasLen = 2;
		Gate g = new Gate(inputSize,hiddenSize,biasLen,new Sigmoid(),new Momentum());
		INDArray w = Nd4j.create(new double[][] {{0.7,0.5},{0.4,0.1}});
		INDArray b = Nd4j.create(new double[][] {{0.3},{0.5}});
		INDArray u = w.dup();
		INDArray h = Nd4j.zeros(b.shape());
		INDArray[] xset = new INDArray[setSize];
		INDArray[] yset = new INDArray[setSize];
		for(int i =0;i<xset.length;i++) {
			xset[i]=Nd4j.rand(new int[] {inputSize,1});
			yset[i] = g.forward(xset[i], h, w, u, b);
		}
		for(int i =0;i<batchTime;i++) {
			
			Set<Integer> iset = new HashSet<Integer>();
			while(iset.size()<batchSize) {
				int rand = (int)(Math.random()*setSize);
				if(rand>=setSize)rand--;
				iset.add(rand);
			}
			INDArray[] xsample = new INDArray[batchSize];
			INDArray[] ypredict = new INDArray[batchSize];
			INDArray[] ylabel = new INDArray[batchSize];
			INDArray[] harray = new INDArray[batchSize];
			INDArray[] acc = new INDArray[batchSize];
			int k = 0;
			double loss  = 0;
			for(Integer node: iset) {
				xsample[k]=xset[node];
				ypredict[k]=g.forward(xsample[k], h);
				ylabel[k]=yset[node];
				harray[k]=Nd4j.zeros(new int[] {hiddenSize,1});
				acc[k]=ypredict[k].sub(ylabel[k]);
				loss += acc[k].norm2Number().doubleValue();
				k++;
			}
			loss/=k;
			Gradients up = g.backward(xsample, harray, acc);
			
			//System.out.println(g.status());
			//g.printUpdater(up);
			g.update(up);
			if(i%100==0) {
				System.out.println("loop i="+i);
				System.out.println("loss :"+loss);
				System.out.println(g.status());
			}
		}
		
	}
	@Test
	public void testGate() {
		int setSize = 1000;
		int batchSize = 30;
		int batchTime = 2900;
		int inputSize = 1;
		int hiddenSize = 1;
		int biasLen = 1;
		Gate g = new Gate(inputSize,hiddenSize,biasLen,new Sigmoid(),new SGD());
		INDArray w = Nd4j.create(new double[] {0.7});
		INDArray b = Nd4j.create(new double[] {0.3});
		INDArray u = Nd4j.create(new double[] {0});
		INDArray h = Nd4j.create(new double[] {0});
		INDArray[] xset = new INDArray[setSize];
		INDArray[] yset = new INDArray[setSize];
		for(int i =0;i<xset.length;i++) {
			xset[i]=Nd4j.create(new double[] {Math.random()*10});
			yset[i] = g.forward(xset[i], h, w, u, b);
		}
		for(int i =0;i<batchTime;i++) {
			
			Set<Integer> iset = new HashSet<Integer>();
			while(iset.size()<batchSize) {
				int rand = (int)(Math.random()*setSize);
				if(rand>=setSize)rand--;
				iset.add(rand);
			}
			INDArray[] xsample = new INDArray[batchSize];
			INDArray[] ypredict = new INDArray[batchSize];
			INDArray[] ylabel = new INDArray[batchSize];
			INDArray[] harray = new INDArray[batchSize];
			INDArray[] acc = new INDArray[batchSize];
			int k = 0;
			double loss  = 0;
			for(Integer node: iset) {
				xsample[k]=xset[node];
				ypredict[k]=g.forward(xsample[k], h);
				ylabel[k]=yset[node];
				harray[k]=Nd4j.zeros(hiddenSize);
				acc[k]=ypredict[k].sub(ylabel[k]);
				loss += acc[k].norm2Number().doubleValue();
				k++;
			}
			loss/=k;
			Gradients up = g.backward(xsample, harray, acc);
			
			//System.out.println(g.status());
			//g.printUpdater(up);
			g.update(up);
			if(i%100==0) {
				System.out.println("loop i="+i);
				System.out.println("loss :"+loss);
				System.out.println(g.status());
			}
		}
		
	}

}
