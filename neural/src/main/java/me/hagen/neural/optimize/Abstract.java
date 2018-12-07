package me.hagen.neural.optimize;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Abstract {
	
	public static Gradients getAverage(Gradients[] g) {
		if(g.length==0)return null;
		Gradients avg = new Gradients(g[0].delta.length);
		for(int i = 0;i<avg.delta.length;i++) {
			avg.delta[i]=Nd4j.zeros(g[0].delta[i].shape());
		}
		for(int i = 0;i<g.length;i++) {
			for(int j = 0;j<g[0].delta.length;j++) {
				avg.delta[j].addi(g[i].delta[j]);
			}
		}
		for(int i = 0;i<avg.delta.length;i++) {
			avg.delta[i].muli(1.0/g.length);
		}
		return avg;
	}
	public static class Gradients{
		
		public Gradients(int size) {
			delta  = new INDArray[size];
		}
		public INDArray delta[];
	}
	// Gradients History
	public static interface GradientFields {
		public void updateFields(Gradients  grad);
	}
	
	//
	public static class LearningRate{
		public double[] rate;
		public LearningRate(int size, double init) {
			rate = new double[size];
			for(int i = 0;i<size;i++) {
				rate[i]=init;
			}
		}
	}
	
	public static interface Optimizer{
		
		public Gradients getDeltaGradient(GradientFields fields, Gradients grad, LearningRate r);
		
		public GradientFields getFields(int paramSize);
		
		public LearningRate getLearningRate(int paramSize,double init);
	}
}
