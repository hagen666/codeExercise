package me.hagen.neural.optimize;

import org.nd4j.linalg.api.ndarray.INDArray;

import me.hagen.neural.optimize.Abstract.GradientFields;
import me.hagen.neural.optimize.Abstract.Gradients;
import me.hagen.neural.optimize.Abstract.Optimizer;
import me.hagen.neural.optimize.Abstract.LearningRate;

public class Adam implements Optimizer {

	public double estim = 0.000001;
	@Override
	public Gradients getDeltaGradient(GradientFields fields, Gradients grad, LearningRate r) {
		// TODO Auto-generated method stub
		fields.updateFields(grad);
		AdamFields af = (AdamFields) fields;
		Gradients g = new Gradients(grad.delta.length);
		for(int i = 0;i<grad.delta.length;i++) {
			INDArray m = af.m[i].mul(1.0/(1-af.miu));
			double n = af.n[i]/(1-af.v);
			g.delta[i]=m.mul(r.rate[i]/(Math.sqrt(n)+estim));
		}
		return g;
	}

	@Override
	public GradientFields getFields(int paramSize) {
		return new AdamFields(paramSize);
	}

	@Override
	public LearningRate getLearningRate(int paramSize, double init) {
		return new LearningRate(paramSize,init);
	}

	public static class AdamFields implements GradientFields{
		
		public INDArray m[];
		public double n[];
		public double miu = 0.9;
		public double v = 0.999;
		public AdamFields(int size) {
			m = new INDArray[size];
			n = new double[size];
		}
		@Override
		public void updateFields(Gradients grad) {
			for(int i = 0;i<grad.delta.length;i++) {
				n[i]+= v*n[i]+(1-v)*grad.delta[i].norm2Number().doubleValue();
				if(m[i]==null) {
					m[i]=grad.delta[i].mul(1-miu);
				}else {
					m[i].muli(miu);
					m[i].addi(grad.delta[i].mul(1-miu));
				}
			}
			
		}
		
	}
}
